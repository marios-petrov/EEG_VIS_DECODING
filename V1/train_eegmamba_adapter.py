#!/usr/bin/env python3
"""
Train EEG-to-Video Adapter using EEGMamba + SD 3.5
========================================================
Trains ONLY the adapter (EEGMamba + projection head)
SD 3.5 transformer, VAE, and text encoders are FROZEN
"""

# CRITICAL: Set cache to /local-scratch BEFORE any imports
import os
import sys

# Add EEGMamba to Python path
sys.path.insert(0, '/local-scratch/marios-datasets/EEGMamba')

os.environ['HF_HOME'] = '/local-scratch/marios-datasets/hf-cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/local-scratch/marios-datasets/hf-cache/hub'
os.environ['TRANSFORMERS_CACHE'] = '/local-scratch/marios-datasets/hf-cache/transformers'
os.environ['TORCH_HOME'] = '/local-scratch/marios-datasets/hf-cache/torch'
os.environ['TMPDIR'] = '/local-scratch/marios-datasets/tmp'
os.environ['TEMP'] = '/local-scratch/marios-datasets/tmp'
os.environ['TMP'] = '/local-scratch/marios-datasets/tmp'

import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from eegmamba_adapter import EEGMambaAdapter
from models_sd35 import load_diffusion_models


class EEGVideoDataset(Dataset):
    """
    Dataset for training EEG-to-Video adapter
    Loads EEG, video latents, and captions
    """
    
    def __init__(
        self,
        preprocessed_dir: Path,
        task: str,
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.task = task
        
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
        self.total_samples = self.num_segments * self.frames_per_segment
        
        print(f"✓ Dataset loaded:")
        print(f"   Segments: {self.num_segments:,}")
        print(f"   Frames per segment: {self.frames_per_segment}")
        print(f"   Total samples: {self.total_samples:,}")
        print(f"   EEG shape per segment: {self.eeg_mmap.shape[1:]}")
        print(f"   Latent shape per frame: {self.latents_mmap.shape[2:]}")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Calculate segment and frame indices
        seg_idx = idx // self.frames_per_segment
        frame_idx = idx % self.frames_per_segment
        
        # Load EEG (same for all frames in segment)
        eeg = self.eeg_mmap[seg_idx].copy()
        eeg = torch.from_numpy(eeg).float()  # [C, T]
        
        # Load video latent
        latent = self.latents_mmap[seg_idx, frame_idx].copy()
        latent = torch.from_numpy(latent).float()  # [16, 96, 96]
        
        # Get caption
        caption = self.captions[seg_idx] if seg_idx < len(self.captions) else ""
        
        return {
            'eeg': eeg,
            'latent': latent,
            'caption': caption,
            'subject_id': torch.tensor(seg_idx % 22, dtype=torch.long),  # Assuming 22 subjects
        }


def encode_prompt_sd3(prompts, text_encoders, tokenizers, device):
    """Encode prompts using SD 3.5's 3 text encoders"""
    
    # CLIP-L
    inputs_l = tokenizers['clip_l'](
        prompts, padding="max_length", max_length=77,
        truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs_l = text_encoders['clip_l'](**inputs_l)
        embeds_l = outputs_l.last_hidden_state
        pooled_l = outputs_l.pooler_output  # [B, 768]
    
    # CLIP-G  
    inputs_g = tokenizers['clip_g'](
        prompts, padding="max_length", max_length=77,
        truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs_g = text_encoders['clip_g'](**inputs_g)
        embeds_g = outputs_g.last_hidden_state
        pooled_g = outputs_g.pooler_output  # [B, 1280]
    
    # T5
    inputs_t5 = tokenizers['t5'](
        prompts, padding="max_length", max_length=256,
        truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        embeds_t5 = text_encoders['t5'](**inputs_t5).last_hidden_state
    
    # Concatenate along sequence dimension
    # CLIP-L: [B, 77, 768]
    # CLIP-G: [B, 77, 1280]  
    # T5: [B, 256, 4096]
    
    # Pad CLIP embeddings to match T5's hidden size (4096)
    B = embeds_l.shape[0]
    
    # Pad CLIP-L: [B, 77, 768] -> [B, 77, 4096]
    embeds_l_padded = torch.nn.functional.pad(
        embeds_l, (0, 4096 - 768), mode='constant', value=0
    )
    
    # Pad CLIP-G: [B, 77, 1280] -> [B, 77, 4096]
    embeds_g_padded = torch.nn.functional.pad(
        embeds_g, (0, 4096 - 1280), mode='constant', value=0
    )
    
    # Concatenate: [B, 77+77+256, 4096]
    prompt_embeds = torch.cat([embeds_l_padded, embeds_g_padded, embeds_t5], dim=1)
    
    # Pooled embeddings: concatenate CLIP-L and CLIP-G pooled outputs
    # [B, 768] + [B, 1280] = [B, 2048]
    pooled_prompt_embeds = torch.cat([pooled_l, pooled_g], dim=-1)
    
    return prompt_embeds, pooled_prompt_embeds


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def main():
    parser = argparse.ArgumentParser(description="Train EEGMamba-SD3.5 Adapter")
    
    # Data arguments
    parser.add_argument('--task', required=True, choices=['dme', 'tp', 'inscapes'])
    parser.add_argument('--preprocessed-dir', required=True,
                       help='Directory with SD 3.5 preprocessed data (16-channel latents)')
    parser.add_argument('--output-dir', 
                       default='/local-scratch/marios-datasets/checkpoints/eegmamba_adapter')
    parser.add_argument('--eegmamba-pretrained',
                       default='/local-scratch/marios-datasets/pretrained_EEGMamba.pth',
                       help='Path to pretrained EEGMamba weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32, help='Per-GPU batch size')
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    
    # Model arguments
    parser.add_argument('--freeze-eegmamba', action='store_true',
                       help='Freeze EEGMamba backbone (only train projection)')
    parser.add_argument('--use-glmnet', action='store_true', default=True,
                       help='Use GLMNet for channel selection')
    parser.add_argument('--num-subjects', type=int, default=22)
    
    # Loss weights
    parser.add_argument('--reconstruction-weight', type=float, default=1.0,
                       help='Weight for reconstruction loss')
    parser.add_argument('--diffusion-weight', type=float, default=0.1,
                       help='Weight for diffusion alignment loss')
    
    # Precision
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp16')
    
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    is_main = (rank == 0)
    
    # Create output directory
    output_dir = Path(args.output_dir) / f"adapter_{args.task}"
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Print header
        print("=" * 70)
        print("TRAINING EEGMAMBA ADAPTER WITH SD 3.5")
        print("=" * 70)
        print(f"✓ Output directory: {output_dir}")
        print(f"✓ GPUs: {world_size}")
        
        # Check disk space
        stat = shutil.disk_usage(output_dir)
        free_gb = stat.free / (1024**3)
        print(f"✓ Free space: {free_gb:.1f} GB")
        print("=" * 70)
        print()
    
    # Load dataset
    if is_main:
        print("📊 Loading dataset...")
    
    dataset = EEGVideoDataset(
        preprocessed_dir=Path(args.preprocessed_dir),
        task=args.task,
    )
    
    # Get EEG dimensions from dataset
    in_channels = dataset.eeg_mmap.shape[1]  # Number of EEG channels
    eeg_time_steps = dataset.eeg_mmap.shape[2]  # Time steps per segment
    
    # Create dataloader with distributed sampler
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    ) if world_size > 1 else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    if is_main:
        print(f"  ✓ Dataset size: {len(dataset):,}")
        print(f"  ✓ Batches per GPU: {len(dataloader)}")
        print(f"  ✓ Effective batch size: {args.batch_size * world_size * args.grad_accum}")
        print()
    
    # Load SD 3.5 models (frozen)
    if is_main:
        print("🎨 Loading SD 3.5 models (frozen)...")
    
    vae, transformer, text_encoders, tokenizers, scheduler = load_diffusion_models(
        model_name="stabilityai/stable-diffusion-3.5-medium",
        model_type="sd3.5",
        device=device,
        load_vae=False,  # Don't need VAE for training
        load_transformer=True,
        load_text_encoders=True,
    )
    
    if is_main:
        print("  ✓ SD 3.5 models loaded and frozen")
        print()
    
    # Create EEGMamba adapter
    if is_main:
        print("🧠 Creating EEGMamba adapter...")
    
    adapter = EEGMambaAdapter(
        in_channels=in_channels,
        eeg_time_steps=eeg_time_steps,
        latent_channels=16,  # SD 3.5
        latent_height=96,
        latent_width=96,
        freeze_eegmamba=args.freeze_eegmamba,
        eegmamba_pretrained_path=args.eegmamba_pretrained,
        use_glmnet=args.use_glmnet,
        num_subjects=args.num_subjects,
    ).to(device)
    
    if is_main:
        trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in adapter.parameters())
        print(f"  ✓ Adapter created")
        print(f"  ✓ Total parameters: {total_params:,}")
        print(f"  ✓ Trainable parameters: {trainable_params:,}")
    
    # Wrap with DDP
    if world_size > 1:
        adapter = DDP(adapter, device_ids=[local_rank], output_device=local_rank)
        if is_main:
            print(f"  ✓ DDP enabled across {world_size} GPUs")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # Mixed precision
    scaler = None
    amp_dtype = torch.float32
    
    if args.precision == 'fp16':
        scaler = torch.cuda.amp.GradScaler()
        amp_dtype = torch.float16
        if is_main:
            print("  ✓ FP16 mixed precision enabled")
    elif args.precision == 'bf16':
        amp_dtype = torch.bfloat16
        if is_main:
            print("  ✓ BF16 mixed precision enabled")
    
    # Training loop
    if is_main:
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"Training adapter only (SD 3.5 is frozen)")
        print(f"Epochs: {args.epochs}")
        print(f"Effective batch size: {args.batch_size * world_size * args.grad_accum}")
        print()
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        adapter.train()
        epoch_losses = {'total': 0.0, 'reconstruction': 0.0, 'diffusion': 0.0}
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:03d}", disable=not is_main)
        
        for i, batch in enumerate(pbar):
            eeg = batch['eeg'].to(device, non_blocking=True)
            latents_gt = batch['latent'].to(device, non_blocking=True)
            captions = batch['caption']
            subject_ids = batch['subject_id'].to(device, non_blocking=True)
            
            # Encode captions with SD 3.5 text encoders
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = encode_prompt_sd3(
                    captions, text_encoders, tokenizers, device
                )
            
            # Predict latent from EEG (in FP32 - EEGMamba FFT needs this)
            latents_pred = adapter(eeg, subject_ids)
            
            # Forward pass with mixed precision (for SD 3.5 operations only)
            with torch.autocast(
                device_type='cuda',
                dtype=amp_dtype,
                enabled=(args.precision != 'fp32'),
            ):
                # Loss 1: Reconstruction loss (L2 with ground truth)
                loss_recon = F.mse_loss(latents_pred, latents_gt)
                
                # Loss 2: Diffusion alignment loss (optional)
                # Add noise and predict it with frozen transformer
                loss_diff = torch.tensor(0.0, device=device)
                if args.diffusion_weight > 0:
                    # Sample random timestep
                    timesteps = torch.randint(
                        0, scheduler.config.num_train_timesteps,
                        (latents_pred.shape[0],), device=device
                    )
                    
                    # Add noise to predicted latent using Flow Matching interpolation
                    # Flow matching: x_t = (1-t)*x_0 + t*x_1
                    # where x_0 is the target latent, x_1 is noise, t is normalized timestep
                    noise = torch.randn_like(latents_pred)
                    t = timesteps.float() / scheduler.config.num_train_timesteps  # Normalize to [0, 1]
                    t = t.view(-1, 1, 1, 1)  # Reshape for broadcasting
                    noisy_latents = (1 - t) * latents_pred + t * noise
                    
                    # Predict noise with frozen transformer
                    with torch.no_grad():
                        noise_pred = transformer(
                            noisy_latents,
                            timestep=timesteps,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_prompt_embeds,
                        ).sample
                    
                    loss_diff = F.mse_loss(noise_pred, noise)
                
                # Total loss
                loss = (
                    args.reconstruction_weight * loss_recon +
                    args.diffusion_weight * loss_diff
                )
                loss = loss / args.grad_accum
            
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step with gradient accumulation
            if (i + 1) % args.grad_accum == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            
            # Update metrics
            epoch_losses['total'] += loss.item() * args.grad_accum
            epoch_losses['reconstruction'] += loss_recon.item()
            if args.diffusion_weight > 0:
                epoch_losses['diffusion'] += loss_diff.item()
            
            if is_main:
                pbar.set_postfix(
                    loss=f"{loss.item() * args.grad_accum:.4f}",
                    recon=f"{loss_recon.item():.4f}"
                )
        
        # Epoch metrics
        if is_main:
            avg_loss = epoch_losses['total'] / len(dataloader)
            avg_recon = epoch_losses['reconstruction'] / len(dataloader)
            
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"  Total loss: {avg_loss:.4f}")
            print(f"  Reconstruction: {avg_recon:.4f}")
            if args.diffusion_weight > 0:
                avg_diff = epoch_losses['diffusion'] / len(dataloader)
                print(f"  Diffusion: {avg_diff:.4f}")
            
            # Save checkpoint
            print(f"  💾 Saving checkpoint...")
            
            model_to_save = adapter.module if world_size > 1 else adapter
            
            checkpoint = {
                'adapter': model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler_lr.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'loss': avg_loss,
                'config': {
                    'in_channels': in_channels,
                    'eeg_time_steps': eeg_time_steps,
                    'latent_channels': 16,
                    'num_subjects': args.num_subjects,
                }
            }
            
            torch.save(checkpoint, output_dir / f"adapter_ep{epoch:03d}.pth")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(checkpoint, output_dir / "adapter_best.pth")
                print(f"  ✓ Best model saved (loss: {best_loss:.4f})")
            
            print()
        
        # Step LR scheduler
        scheduler_lr.step()
    
    if is_main:
        print("=" * 70)
        print("✅ TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Checkpoints saved to: {output_dir}")
        print(f"Best loss: {best_loss:.4f}")
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
