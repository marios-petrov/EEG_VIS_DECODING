#!/usr/bin/env python3
"""
Train EEGMamba adapter on SEED-DV dataset for comparison with EEG2Video paper.

Train/test split (matches paper):
- Train: Blocks 1-6 (1200 segments per subject)
- Test: Block 7 (200 segments per subject) - NOT loaded during training

Usage:
    torchrun --nproc_per_node=8 train_eegmamba_seed_dv.py \
        --preprocessed-dir /local-scratch/marios-datasets/SEED/preprocessed_sd35 \
        --output-dir /local-scratch/marios-datasets/SEED/checkpoints/eegmamba_sd35 \
        --epochs 100 \
        --batch-size 16
"""

import os
os.environ.setdefault('HF_HOME', '/home/mpetrov/.cache/huggingface')

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


class SEEDDVTrainDataset(Dataset):
    """
    SEED-DV Training Dataset - ONLY loads train data (blocks 1-6)
    Memory-efficient: uses memory-mapped arrays
    """
    
    def __init__(
        self,
        preprocessed_dir: Path,
        subject_ids: List[int],
        caption_embeddings: dict = None,
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.caption_embeddings = caption_embeddings or {}
        
        # Build list of all training samples
        self.samples = []  # List of (subject_id, segment_idx, is_train)
        self.subject_data = {}  # Lazy-loaded memory maps
        
        for subj_id in subject_ids:
            subj_dir = self.preprocessed_dir / f"sub{subj_id}"
            if not subj_dir.exists():
                continue
            
            # Load metadata to get train/test split
            meta_path = subj_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    metadata = json.load(f)
                segments = metadata.get('segments', [])
                
                # Only add TRAINING segments
                for seg in segments:
                    if seg.get('is_train', True):  # Block 7 has is_train=False
                        self.samples.append({
                            'subject_id': subj_id,
                            'segment_idx': seg['segment_idx'],
                            'caption': seg.get('caption', ''),
                        })
        
        print(f"  ✓ Loaded {len(self.samples):,} TRAINING samples from {len(subject_ids)} subjects")
    
    def _get_subject_data(self, subj_id: int):
        """Lazy load subject data as memory-mapped arrays"""
        if subj_id not in self.subject_data:
            subj_dir = self.preprocessed_dir / f"sub{subj_id}"
            self.subject_data[subj_id] = {
                'eeg': np.load(subj_dir / "eeg.npy", mmap_mode='r'),
                'latents': np.load(subj_dir / "latents.npy", mmap_mode='r'),
            }
        return self.subject_data[subj_id]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        subj_id = sample['subject_id']
        seg_idx = sample['segment_idx']
        caption = sample['caption']
        
        # Get memory-mapped data
        data = self._get_subject_data(subj_id)
        
        # Load EEG
        eeg = torch.from_numpy(data['eeg'][seg_idx].copy()).float()
        
        # Load middle frame latent (frame 3 of 6)
        latent = torch.from_numpy(data['latents'][seg_idx, 3].copy()).float()
        
        # Get caption embedding
        if caption in self.caption_embeddings:
            caption_emb = self.caption_embeddings[caption]
        else:
            caption_emb = torch.zeros(2048)
        
        return {
            'eeg': eeg,
            'latent': latent,
            'caption_emb': caption_emb,
            'subject_id': subj_id,
        }


def precompute_caption_embeddings(preprocessed_dir: Path, subject_ids: List[int], device: torch.device):
    """Precompute CLIP embeddings for all unique captions"""
    from transformers import CLIPTokenizer, CLIPTextModel
    
    print("📝 Collecting unique captions...")
    captions = set()
    
    for subj_id in subject_ids:
        meta_path = preprocessed_dir / f"sub{subj_id}" / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
            for seg in metadata.get('segments', []):
                if seg.get('is_train', True):
                    captions.add(seg.get('caption', ''))
    
    captions = list(captions)
    print(f"  Found {len(captions)} unique captions")
    
    # Load CLIP
    print("📝 Loading CLIP text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    text_encoder.eval()
    
    embeddings = {}
    
    print("📝 Encoding captions...")
    with torch.no_grad():
        for i in tqdm(range(0, len(captions), 32), desc="Encoding"):
            batch = captions[i:i+32]
            
            inputs = tokenizer(
                batch,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            
            outputs = text_encoder(**inputs)
            pooled = outputs.pooler_output  # [B, 768]
            
            # Pad to 2048 for compatibility
            padded = F.pad(pooled, (0, 2048 - 768))
            
            for j, caption in enumerate(batch):
                embeddings[caption] = padded[j].cpu()
    
    # Add empty caption
    embeddings[''] = torch.zeros(2048)
    
    # Cleanup
    del text_encoder, tokenizer
    torch.cuda.empty_cache()
    
    print(f"✓ Computed {len(embeddings)} caption embeddings")
    return embeddings


def train_epoch(model, dataloader, optimizer, scaler, device, epoch, rank):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=(rank != 0))
    
    for batch in pbar:
        eeg = batch['eeg'].to(device, non_blocking=True)
        target = batch['latent'].to(device, non_blocking=True)
        caption_emb = batch['caption_emb'].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            pred = model(eeg, caption_emb)
            loss = F.mse_loss(pred, target)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        if rank == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--subjects', type=str, default='1-20')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-checkpoints', action='store_true',
                       help='Save periodic checkpoints (default: only best)')
    args = parser.parse_args()
    
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    is_main = (rank == 0)
    
    # Parse subjects
    if '-' in args.subjects:
        start, end = map(int, args.subjects.split('-'))
        subject_ids = list(range(start, end + 1))
    else:
        subject_ids = [int(s) for s in args.subjects.split(',')]
    
    preprocessed_dir = Path(args.preprocessed_dir)
    output_dir = Path(args.output_dir)
    
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 60)
        print("SEED-DV EEGMamba Training")
        print("=" * 60)
        print(f"Subjects: {subject_ids}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size} x {world_size} GPUs = {args.batch_size * world_size}")
        print(f"Output: {output_dir}")
    
    # Precompute caption embeddings (only on main, then load on others)
    embeddings_path = output_dir / "caption_embeddings.pt"
    
    if is_main:
        if embeddings_path.exists():
            print(f"\n📂 Loading cached caption embeddings...")
            caption_embeddings = torch.load(embeddings_path)
        else:
            caption_embeddings = precompute_caption_embeddings(
                preprocessed_dir, subject_ids, device
            )
            torch.save(caption_embeddings, embeddings_path)
    
    if world_size > 1:
        dist.barrier()
    
    if not is_main:
        caption_embeddings = torch.load(embeddings_path)
    
    # Create dataset
    if is_main:
        print(f"\n📂 Creating training dataset...")
    
    train_dataset = SEEDDVTrainDataset(
        preprocessed_dir=preprocessed_dir,
        subject_ids=subject_ids,
        caption_embeddings=caption_embeddings,
    )
    
    # Create dataloader
    sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    
    # Get dimensions from first sample
    sample = train_dataset[0]
    eeg_channels = sample['eeg'].shape[0]
    eeg_samples = sample['eeg'].shape[1]
    latent_channels = sample['latent'].shape[0]
    latent_height = sample['latent'].shape[1]
    latent_width = sample['latent'].shape[2]
    
    if is_main:
        print(f"\n📊 Data dimensions:")
        print(f"  EEG: ({eeg_channels}, {eeg_samples})")
        print(f"  Latent: ({latent_channels}, {latent_height}, {latent_width})")
        print(f"  Training samples: {len(train_dataset):,}")
    
    # Import and create model
    from eegmamba_adapter_optimized import EEGMambaAdapter
    
    model = EEGMambaAdapter(
        eeg_channels=eeg_channels,
        eeg_samples=eeg_samples,
        d_model=512,
        n_layers=4,
        latent_channels=latent_channels,
        latent_height=latent_height,
        latent_width=latent_width,
        text_embed_dim=2048,
    ).to(device)
    
    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n🔧 Model parameters: {total_params:,}")
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    scaler = GradScaler()
    
    # Training loop
    best_loss = float('inf')
    best_epoch = 0
    
    if is_main:
        print(f"\n{'='*60}")
        print("Starting training...")
        print(f"{'='*60}\n")
    
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch, rank
        )
        
        scheduler.step()
        
        # Sync loss across GPUs
        if world_size > 1:
            loss_tensor = torch.tensor([train_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            train_loss = loss_tensor.item() / world_size
        
        if is_main:
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.6f} | LR: {lr:.2e}")
            
            # Save best model (ALWAYS)
            if train_loss < best_loss:
                best_loss = train_loss
                best_epoch = epoch
                
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': epoch,
                    'model': model_to_save.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': train_loss,
                    'config': {
                        'eeg_channels': eeg_channels,
                        'eeg_samples': eeg_samples,
                        'latent_channels': latent_channels,
                        'latent_height': latent_height,
                        'latent_width': latent_width,
                    }
                }, output_dir / "adapter_best.pt")
                print(f"  → Saved best model (loss: {best_loss:.6f})")
            
            # Periodic checkpoint (only if flag set)
            if args.save_checkpoints and epoch % 10 == 0:
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': epoch,
                    'model': model_to_save.state_dict(),
                    'loss': train_loss,
                }, output_dir / f"adapter_epoch{epoch:03d}.pt")
    
    if is_main:
        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"Best loss: {best_loss:.6f} (epoch {best_epoch})")
        print(f"Checkpoint: {output_dir / 'adapter_best.pt'}")
        print(f"{'='*60}")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
