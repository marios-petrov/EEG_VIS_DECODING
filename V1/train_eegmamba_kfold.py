#!/usr/bin/env python3
"""
Train EEGMamba Adapter with K-Fold Cross-Validation (Subject-Level Splits)

5-Fold CV with ~80/20 train/test split across 22 subjects:
  Fold 0: test [0-4]   (5 subjects), train [5-21]  (17 subjects)
  Fold 1: test [5-9]   (5 subjects), train [0-4, 10-21] (17 subjects)
  Fold 2: test [10-13] (4 subjects), train [0-9, 14-21] (18 subjects)
  Fold 3: test [14-17] (4 subjects), train [0-13, 18-21] (18 subjects)
  Fold 4: test [18-21] (4 subjects), train [0-17] (18 subjects)

Usage:
    # Train fold 0
    torchrun --nproc_per_node=8 train_eegmamba_kfold.py \
        --fold 0 \
        --preprocessed-dir /path/to/data \
        --output-dir /path/to/checkpoints/fold0
    
    # Train all folds (run 5 times with --fold 0,1,2,3,4)
"""

import os
os.environ['HF_HOME'] = '/local-scratch/marios-datasets/hf-cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/local-scratch/marios-datasets/hf-cache/hub'

import sys
sys.path.insert(0, '/local-scratch/marios-datasets/EEGMamba')

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from eegmamba_adapter_optimized import EEGMambaAdapter
from models_sd35 import load_diffusion_models


def get_fold_subjects(fold: int, num_folds: int = 5, num_subjects: int = 22):
    """
    Get train and test subject IDs for a given fold.
    
    5-fold split for 22 subjects (~80/20):
      Fold 0: test [0-4], train [5-21]
      Fold 1: test [5-9], train [0-4, 10-21]
      Fold 2: test [10-13], train [0-9, 14-21]
      Fold 3: test [14-17], train [0-13, 18-21]
      Fold 4: test [18-21], train [0-17]
    """
    assert 0 <= fold < num_folds, f"Fold must be 0-{num_folds-1}"
    
    # Define test subjects for each fold
    fold_test_subjects = {
        0: list(range(0, 5)),     # [0,1,2,3,4] - 5 subjects
        1: list(range(5, 10)),    # [5,6,7,8,9] - 5 subjects
        2: list(range(10, 14)),   # [10,11,12,13] - 4 subjects
        3: list(range(14, 18)),   # [14,15,16,17] - 4 subjects
        4: list(range(18, 22)),   # [18,19,20,21] - 4 subjects
    }
    
    test_subjects = set(fold_test_subjects[fold])
    train_subjects = set(range(num_subjects)) - test_subjects
    
    return sorted(list(train_subjects)), sorted(list(test_subjects))


class EEGVideoDatasetKFold(Dataset):
    """Dataset with subject-level filtering for k-fold CV"""
    def __init__(
        self,
        preprocessed_dir: Path,
        task: str,
        subject_ids: list,  # Only include these subjects
        num_total_subjects: int = 22,
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.task = task
        self.subject_ids = set(subject_ids)
        self.num_total_subjects = num_total_subjects
        
        # Memory-map EEG data
        eeg_path = preprocessed_dir / f"{task}_eeg.npy"
        print(f"📂 Loading EEG from: {eeg_path}")
        self.eeg_mmap = np.load(eeg_path, mmap_mode='r')
        
        # Memory-map video latents (16-channel for SD 3.5)
        latents_path = preprocessed_dir / f"{task}_vae_latents_hd.npy"
        print(f"📂 Loading latents from: {latents_path}")
        self.latents_mmap = np.load(latents_path, mmap_mode='r')
        
        # Load captions
        captions_path = preprocessed_dir / f"{task}_captions_hd.json"
        if captions_path.exists():
            with open(captions_path, 'r') as f:
                data = json.load(f)
                self.captions = data.get('captions', [])
        else:
            self.captions = [""] * len(self.eeg_mmap)
        
        # Get dimensions
        self.num_segments = self.eeg_mmap.shape[0]
        self.frames_per_segment = self.latents_mmap.shape[1]
        
        # Build index of valid samples (only from specified subjects)
        self.valid_indices = []
        for segment_idx in range(self.num_segments):
            subject_id = segment_idx % num_total_subjects
            if subject_id in self.subject_ids:
                for frame_idx in range(self.frames_per_segment):
                    global_idx = segment_idx * self.frames_per_segment + frame_idx
                    self.valid_indices.append(global_idx)
        
        print(f"✓ Dataset loaded (filtered by subjects):")
        print(f"   Total segments: {self.num_segments:,}")
        print(f"   Frames per segment: {self.frames_per_segment}")
        print(f"   Subjects included: {sorted(self.subject_ids)}")
        print(f"   Valid samples: {len(self.valid_indices):,} / {self.num_segments * self.frames_per_segment:,}")
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Map to actual index
        actual_idx = self.valid_indices[idx]
        
        segment_idx = actual_idx // self.frames_per_segment
        frame_idx = actual_idx % self.frames_per_segment
        
        # Load data
        eeg = torch.from_numpy(self.eeg_mmap[segment_idx].copy()).float()
        latent = torch.from_numpy(self.latents_mmap[segment_idx, frame_idx].copy()).float()
        
        # Caption
        if segment_idx < len(self.captions):
            caption = self.captions[segment_idx]
        else:
            caption = ""
        
        # Subject ID
        subject_id = segment_idx % self.num_total_subjects
        
        return {
            'eeg': eeg,
            'latent': latent,
            'caption': caption,
            'subject_id': subject_id,
            'index': idx,  # Index into valid_indices for embedding lookup
            'global_index': actual_idx,  # Original dataset index
        }


def encode_prompt_sd3(
    prompt: str,
    text_encoders: dict,
    tokenizers: dict,
    device: torch.device,
    max_length: int = 77,
) -> dict:
    """Encode prompt with SD 3.5's three text encoders"""
    # CLIP-L
    inputs_l = tokenizers['clip_l'](
        prompt, padding='max_length', max_length=max_length,
        truncation=True, return_tensors='pt',
    ).to(device)
    outputs_l = text_encoders['clip_l'](**inputs_l)
    pooled_l = outputs_l.pooler_output
    
    # CLIP-G
    inputs_g = tokenizers['clip_g'](
        prompt, padding='max_length', max_length=max_length,
        truncation=True, return_tensors='pt',
    ).to(device)
    outputs_g = text_encoders['clip_g'](**inputs_g)
    pooled_g = outputs_g.pooler_output
    
    # T5-XXL
    inputs_t5 = tokenizers['t5'](
        prompt, padding='max_length', max_length=max_length,
        truncation=True, return_tensors='pt',
    ).to(device)
    outputs_t5 = text_encoders['t5'](**inputs_t5)
    hidden_t5 = outputs_t5.last_hidden_state
    
    prompt_embeds = hidden_t5.squeeze(0)
    pooled_prompt_embeds = torch.cat([pooled_l, pooled_g], dim=-1).squeeze(0)
    
    return {
        'prompt_embeds': prompt_embeds,
        'pooled_embeds': pooled_prompt_embeds,
    }


def precompute_text_embeddings(
    dataset: EEGVideoDatasetKFold,
    text_encoders: dict,
    tokenizers: dict,
    device: torch.device,
    save_path: Path,
    batch_size: int = 64,
) -> dict:
    """Pre-compute text embeddings for the filtered dataset"""
    print("\n" + "="*70)
    print("PRE-COMPUTING TEXT EMBEDDINGS")
    print("="*70)
    
    if save_path.exists():
        print(f"\n📂 Loading cached embeddings from {save_path}")
        return torch.load(save_path, weights_only=False)
    
    # Get unique captions from valid segments only
    valid_segment_indices = set()
    for idx in dataset.valid_indices:
        seg_idx = idx // dataset.frames_per_segment
        valid_segment_indices.add(seg_idx)
    
    valid_captions = []
    for seg_idx in valid_segment_indices:
        if seg_idx < len(dataset.captions):
            valid_captions.append(dataset.captions[seg_idx])
        else:
            valid_captions.append("")
    
    unique_captions = list(set(valid_captions))
    caption_to_idx = {cap: i for i, cap in enumerate(unique_captions)}
    
    print(f"Valid segments: {len(valid_segment_indices):,}")
    print(f"Unique captions: {len(unique_captions):,}")
    
    embeddings = {'prompt_embeds': [], 'pooled_embeds': []}
    
    with torch.no_grad():
        for i in tqdm(range(0, len(unique_captions), batch_size), desc="Encoding"):
            batch_captions = unique_captions[i:i+batch_size]
            batch_embeds = [encode_prompt_sd3(cap, text_encoders, tokenizers, device) 
                          for cap in batch_captions]
            
            embeddings['prompt_embeds'].append(
                torch.stack([e['prompt_embeds'] for e in batch_embeds]).cpu()
            )
            embeddings['pooled_embeds'].append(
                torch.stack([e['pooled_embeds'] for e in batch_embeds]).cpu()
            )
    
    embeddings['prompt_embeds'] = torch.cat(embeddings['prompt_embeds'], dim=0)
    embeddings['pooled_embeds'] = torch.cat(embeddings['pooled_embeds'], dim=0)
    
    # Create index map: dataset idx -> embedding idx
    index_map = []
    for idx in range(len(dataset)):
        actual_idx = dataset.valid_indices[idx]
        seg_idx = actual_idx // dataset.frames_per_segment
        caption = dataset.captions[seg_idx] if seg_idx < len(dataset.captions) else ""
        embed_idx = caption_to_idx[caption]
        index_map.append(embed_idx)
    
    embeddings['index_map'] = torch.tensor(index_map)
    
    print(f"\n✓ Embeddings computed:")
    print(f"  Prompt embeds: {embeddings['prompt_embeds'].shape}")
    print(f"  Pooled embeds: {embeddings['pooled_embeds'].shape}")
    print(f"  Index map: {embeddings['index_map'].shape}")
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, save_path)
    print(f"💾 Saved to: {save_path}")
    
    return embeddings


def train_epoch(
    adapter: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    vae,
    transformer,
    scheduler,
    precomputed_embeds: dict,
    device: torch.device,
    epoch: int,
    args,
    rank: int = 0,
):
    """Train for one epoch"""
    adapter.train()
    
    total_loss = 0
    total_recon_loss = 0
    total_diff_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch:03d}") if rank == 0 else dataloader
    
    for batch_idx, batch in enumerate(pbar):
        eeg = batch['eeg'].to(device)
        latents_gt = batch['latent'].to(device)
        subject_ids = batch['subject_id'].to(device)
        indices = batch['index']
        
        B = eeg.shape[0]
        
        # Get pre-computed embeddings
        embed_indices = precomputed_embeds['index_map'][indices.cpu()]
        prompt_embeds = precomputed_embeds['prompt_embeds'][embed_indices].to(device)
        pooled_embeds = precomputed_embeds['pooled_embeds'][embed_indices].to(device)
        
        with torch.cuda.amp.autocast(dtype=torch.float16):
            latents_pred = adapter(eeg, subject_ids)
            recon_loss = F.mse_loss(latents_pred, latents_gt)
            
            diff_loss = torch.tensor(0.0, device=device)
            if args.diffusion_weight > 0:
                t = torch.rand(B, device=device) * 0.6 + 0.2
                t_expanded = t.view(-1, 1, 1, 1)
                noise = torch.randn_like(latents_pred)
                noisy_latents = (1 - t_expanded) * latents_pred + t_expanded * noise
                timesteps = (t * scheduler.config.num_train_timesteps).long()
                
                with torch.no_grad():
                    model_pred = transformer(
                        hidden_states=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_embeds,
                        return_dict=False,
                    )[0]
                
                target_velocity = noise - latents_pred
                diff_loss = F.mse_loss(model_pred, target_velocity)
            
            loss = args.reconstruction_weight * recon_loss + args.diffusion_weight * diff_loss
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_diff_loss += diff_loss.item()
        
        if rank == 0:
            postfix = {'loss': f"{loss.item():.4f}", 'recon': f"{recon_loss.item():.4f}"}
            if args.diffusion_weight > 0:
                postfix['diff'] = f"{diff_loss.item():.4f}"
            pbar.set_postfix(postfix)
    
    n = len(dataloader)
    return total_loss / n, total_recon_loss / n, total_diff_loss / n


def main():
    parser = argparse.ArgumentParser()
    
    # K-Fold arguments
    parser.add_argument('--fold', type=int, required=True, choices=[0,1,2,3,4],
                       help='Fold number (0-4)')
    parser.add_argument('--num-folds', type=int, default=5)
    
    # Data
    parser.add_argument('--task', type=str, default='dme', choices=['dme', 'tp', 'inscapes'])
    parser.add_argument('--preprocessed-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    
    # Model
    parser.add_argument('--use-glmnet', action='store_true', default=True)
    parser.add_argument('--num-subjects', type=int, default=22)
    parser.add_argument('--freeze-eegmamba', action='store_true')
    parser.add_argument('--eegmamba-pretrained', type=str,
                       default='/local-scratch/marios-datasets/pretrained_EEGMamba.pth')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--reconstruction-weight', type=float, default=1.0)
    parser.add_argument('--diffusion-weight', type=float, default=0.0)
    
    # System
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp32', 'fp16'])
    parser.add_argument('--local_rank', type=int, default=0)
    
    args = parser.parse_args()
    
    # Setup DDP
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    device = torch.device(f'cuda:{local_rank}')
    
    # Get fold split
    train_subjects, test_subjects = get_fold_subjects(args.fold, args.num_folds, args.num_subjects)
    
    if rank == 0:
        print("="*70)
        print(f"TRAINING EEGMAMBA ADAPTER - FOLD {args.fold}/{args.num_folds-1}")
        print("="*70)
        print(f"\n📊 K-Fold Split:")
        print(f"   Train subjects ({len(train_subjects)}): {train_subjects}")
        print(f"   Test subjects ({len(test_subjects)}): {test_subjects}")
        print(f"   Split ratio: {len(train_subjects)}/{len(test_subjects)} "
              f"({100*len(train_subjects)/args.num_subjects:.0f}%/"
              f"{100*len(test_subjects)/args.num_subjects:.0f}%)")
        print(f"\nTask: {args.task}")
        print(f"GPUs: {world_size}")
        print(f"Batch size per GPU: {args.batch_size}")
    
    # Load dataset (filtered to train subjects only)
    preprocessed_dir = Path(args.preprocessed_dir)
    dataset = EEGVideoDatasetKFold(
        preprocessed_dir=preprocessed_dir,
        task=args.task,
        subject_ids=train_subjects,
        num_total_subjects=args.num_subjects,
    )
    
    # Load models
    if rank == 0:
        print("\nLoading SD 3.5 components...")
    
    vae, transformer, text_encoders, tokenizers, scheduler = load_diffusion_models(
        model_name="stabilityai/stable-diffusion-3.5-medium",
        model_type="sd3.5",
        device=device,
        load_vae=True,
        load_transformer=True,
        load_text_encoders=True,
    )
    
    # Pre-compute embeddings (per-fold cache)
    output_dir = Path(args.output_dir)
    embed_cache_path = output_dir / f"precomputed_embeddings_fold{args.fold}.pt"
    precomputed_embeds = precompute_text_embeddings(
        dataset=dataset,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        device=device,
        save_path=embed_cache_path,
    )
    
    del text_encoders, tokenizers
    torch.cuda.empty_cache()
    
    if rank == 0:
        print("\n✓ Text encoders unloaded")
    
    # Create adapter
    adapter = EEGMambaAdapter(
        in_channels=dataset.eeg_mmap.shape[1],
        eeg_time_steps=dataset.eeg_mmap.shape[2],
        latent_height=96,
        latent_width=96,
        latent_channels=16,
        use_glmnet=args.use_glmnet,
        num_subjects=args.num_subjects,
        freeze_eegmamba=args.freeze_eegmamba,
        eegmamba_pretrained_path=args.eegmamba_pretrained,
    ).to(device)
    
    if world_size > 1:
        adapter = DDP(adapter, device_ids=[local_rank])
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda')
    
    sampler = DistributedSampler(dataset) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        drop_last=True,
    )
    
    if rank == 0:
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        print(f"Training samples: {len(dataset):,}")
        print(f"Epochs: {args.epochs}")
        
        total_params = sum(p.numel() for p in adapter.parameters())
        trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        print(f"\nAdapter parameters: {trainable_params:,} / {total_params:,}")
    
    # Save fold info
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "fold_info.json", 'w') as f:
        json.dump({
            'fold': args.fold,
            'num_folds': args.num_folds,
            'train_subjects': train_subjects,
            'test_subjects': test_subjects,
            'num_train_samples': len(dataset),
        }, f, indent=2)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        if sampler:
            sampler.set_epoch(epoch)
        
        avg_loss, avg_recon, avg_diff = train_epoch(
            adapter=adapter,
            dataloader=dataloader,
            optimizer=optimizer,
            scaler=scaler,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            precomputed_embeds=precomputed_embeds,
            device=device,
            epoch=epoch,
            args=args,
            rank=rank,
        )
        
        if rank == 0:
            print(f"\nEpoch {epoch:03d}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}")
            
            # Save best
            if avg_loss < best_loss:
                best_loss = avg_loss
                ckpt_path = output_dir / "adapter_best.pt"
                torch.save({
                    'epoch': epoch,
                    'fold': args.fold,
                    'adapter': adapter.module.state_dict() if world_size > 1 else adapter.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': avg_loss,
                    'train_subjects': train_subjects,
                    'test_subjects': test_subjects,
                }, ckpt_path)
                print(f"  💾 New best! Saved to {ckpt_path}")
            
            # Periodic save
            if epoch % 10 == 0:
                ckpt_path = output_dir / f"adapter_epoch{epoch:03d}.pt"
                torch.save({
                    'epoch': epoch,
                    'fold': args.fold,
                    'adapter': adapter.module.state_dict() if world_size > 1 else adapter.state_dict(),
                    'loss': avg_loss,
                    'train_subjects': train_subjects,
                    'test_subjects': test_subjects,
                }, ckpt_path)
    
    if rank == 0:
        print("\n" + "="*70)
        print(f"FOLD {args.fold} TRAINING COMPLETE!")
        print(f"Best loss: {best_loss:.4f}")
        print(f"Test subjects for evaluation: {test_subjects}")
        print("="*70)
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
