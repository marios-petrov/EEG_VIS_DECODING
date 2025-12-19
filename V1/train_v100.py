#!/usr/bin/env python3
"""
Main training script for EEG2Video adapter
Optimized for 8x V100 32GB GPUs with DistributedDataParallel
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import wandb

from models import EEGAdapter, VideoLatentAdapter, load_diffusion_models
from utils import (
    EEGVideoDataset,
    seed_everything,
    save_checkpoint,
    load_checkpoint,
    print_metrics,
)


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


class EEG2VideoTrainer:
    """Trainer for EEG to video reconstruction"""
    
    def __init__(
        self,
        adapter: nn.Module,
        vae: nn.Module,
        unet: nn.Module,
        noise_scheduler,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        use_semantic_loss: bool = True,
        semantic_loss_weight: float = 0.1,
        use_perceptual_loss: bool = False,
        perceptual_loss_weight: float = 0.1,
        use_wandb: bool = False,
        rank: int = 0,
    ):
        self.adapter = adapter
        self.vae = vae
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.optimizer = optimizer
        self.device = device
        self.rank = rank
        self.is_main = rank == 0
        
        self.use_semantic_loss = use_semantic_loss
        self.semantic_loss_weight = semantic_loss_weight
        self.use_perceptual_loss = use_perceptual_loss
        self.perceptual_loss_weight = perceptual_loss_weight
        self.use_wandb = use_wandb and self.is_main
        
        self.clip_embeddings = None
        
        # Perceptual loss model
        if use_perceptual_loss:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True).features[:16].to(device).eval()
            for p in vgg.parameters():
                p.requires_grad = False
            self.vgg = vgg
    
    def load_semantic_embeddings(self, embeddings_path: Path):
        """Load CLIP text embeddings for semantic guidance"""
        self.clip_embeddings = torch.from_numpy(
            np.load(embeddings_path)
        ).float().to(self.device)
        if self.is_main:
            print(f"✓ Loaded semantic embeddings: {self.clip_embeddings.shape}")
    
    def compute_reconstruction_loss(
        self,
        pred_latent: torch.Tensor,
        gt_latent: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss between predicted and ground truth latents"""
        return F.mse_loss(pred_latent, gt_latent)
    
    def compute_semantic_loss(
        self,
        pred_latent: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Semantic alignment loss using CLIP text embeddings"""
        if self.clip_embeddings is None:
            return torch.tensor(0.0, device=self.device)
        
        # Get corresponding CLIP embeddings
        batch_clip = self.clip_embeddings[indices, 0, 0, :]  # [B, 1024]
        
        # Average pool adapter output
        pred_semantic = F.adaptive_avg_pool2d(pred_latent, (1, 1)).squeeze()  # [B, 4]
        
        # Project to CLIP dimension
        if not hasattr(self, 'semantic_proj'):
            self.semantic_proj = nn.Linear(4, 1024).to(self.device)
        
        pred_semantic = self.semantic_proj(pred_semantic)  # [B, 1024]
        
        # Cosine similarity loss
        loss = 1.0 - F.cosine_similarity(pred_semantic, batch_clip, dim=-1).mean()
        
        return loss
    
    def compute_perceptual_loss(
        self,
        pred_latent: torch.Tensor,
        gt_latent: torch.Tensor,
    ) -> torch.Tensor:
        """VGG perceptual loss in pixel space"""
        if not self.use_perceptual_loss:
            return torch.tensor(0.0, device=self.device)
        
        # Decode latents to pixel space
        with torch.no_grad():
            pred_img = self.vae.decode(pred_latent / self.vae.config.scaling_factor).sample
            gt_img = self.vae.decode(gt_latent / self.vae.config.scaling_factor).sample
        
        # Extract VGG features
        pred_feat = self.vgg(pred_img)
        gt_feat = self.vgg(gt_img)
        
        return F.mse_loss(pred_feat, gt_feat)
    
    def train_step(self, batch: dict) -> dict:
        """Single training step"""
        eeg = batch['eeg'].to(self.device, non_blocking=True)
        gt_latent = batch['latent'][:, 0].to(self.device, non_blocking=True)
        subject_ids = batch['subject_id'].to(self.device, non_blocking=True)
        indices = batch['index']
        
        # Forward pass
        pred_latent = self.adapter(eeg, subject_ids)
        
        # Reconstruction loss
        recon_loss = self.compute_reconstruction_loss(pred_latent, gt_latent)
        
        # Semantic guidance loss
        semantic_loss = torch.tensor(0.0, device=self.device)
        if self.use_semantic_loss and self.clip_embeddings is not None:
            semantic_loss = self.compute_semantic_loss(pred_latent, indices)
        
        # Perceptual loss
        perceptual_loss = torch.tensor(0.0, device=self.device)
        if self.use_perceptual_loss:
            perceptual_loss = self.compute_perceptual_loss(pred_latent, gt_latent)
        
        # Total loss
        total_loss = (
            recon_loss +
            self.semantic_loss_weight * semantic_loss +
            self.perceptual_loss_weight * perceptual_loss
        )
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'semantic_loss': semantic_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
        }
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict:
        """Train for one epoch"""
        self.adapter.train()
        
        epoch_metrics = {
            'loss': 0.0,
            'recon_loss': 0.0,
            'semantic_loss': 0.0,
            'perceptual_loss': 0.0,
        }
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not self.is_main)
        for batch in pbar:
            metrics = self.train_step(batch)
            
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]
            
            if self.is_main:
                pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})
            
            if self.use_wandb:
                wandb.log(metrics)
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= len(dataloader)
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict:
        """Validate on validation set"""
        self.adapter.eval()
        
        val_metrics = {
            'loss': 0.0,
            'recon_loss': 0.0,
            'semantic_loss': 0.0,
        }
        
        for batch in tqdm(dataloader, desc="Validating", disable=not self.is_main):
            eeg = batch['eeg'].to(self.device, non_blocking=True)
            gt_latent = batch['latent'][:, 0].to(self.device, non_blocking=True)
            subject_ids = batch['subject_id'].to(self.device, non_blocking=True)
            indices = batch['index']
            
            pred_latent = self.adapter(eeg, subject_ids)
            
            recon_loss = self.compute_reconstruction_loss(pred_latent, gt_latent)
            semantic_loss = torch.tensor(0.0, device=self.device)
            if self.use_semantic_loss and self.clip_embeddings is not None:
                semantic_loss = self.compute_semantic_loss(pred_latent, indices)
            
            total_loss = recon_loss + self.semantic_loss_weight * semantic_loss
            
            val_metrics['loss'] += total_loss.item()
            val_metrics['recon_loss'] += recon_loss.item()
            val_metrics['semantic_loss'] += semantic_loss.item()
        
        for key in val_metrics:
            val_metrics[key] /= len(dataloader)
        
        return val_metrics


def main():
    parser = argparse.ArgumentParser(description="Train EEG2Video adapter (Multi-GPU)")
    
    # Data arguments
    parser.add_argument('--preprocessed-dir', required=True)
    parser.add_argument('--task', required=True, choices=['dme', 'tp', 'inscapes'])
    parser.add_argument('--output-dir', required=True)
    
    # Model arguments
    parser.add_argument('--adapter-type', default='video', choices=['base', 'video'])
    parser.add_argument('--in-channels', type=int, default=63)
    parser.add_argument('--eeg-time-steps', type=int, default=125)
    parser.add_argument('--hidden-dim', type=int, default=768)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--use-glmnet', action='store_true', default=True)
    parser.add_argument('--num-subjects', type=int, default=22)
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32, help='Per-GPU batch size')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    
    # Loss arguments
    parser.add_argument('--use-semantic-loss', action='store_true', default=True)
    parser.add_argument('--semantic-loss-weight', type=float, default=0.1)
    parser.add_argument('--use-perceptual-loss', action='store_true', default=False)
    parser.add_argument('--perceptual-loss-weight', type=float, default=0.1)
    parser.add_argument('--clip-embeddings-path', help='Path to CLIP embeddings')
    
    # System arguments
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-project', default='eeg2video')
    parser.add_argument('--resume', help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    is_main = rank == 0
    
    # Setup
    seed_everything(args.seed + rank)
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if args.wandb and is_main:
        wandb.init(project=args.wandb_project, config=vars(args))
    
    if is_main:
        print("=" * 70)
        print(f"INITIALIZING TRAINING - {world_size} GPUs")
        print("=" * 70)
    
    # Load datasets
    if is_main:
        print("\n📊 Loading datasets...")
    
    train_dataset = EEGVideoDataset(
        preprocessed_dir=args.preprocessed_dir,
        task=args.task,
        split='train',
    )
    val_dataset = EEGVideoDataset(
        preprocessed_dir=args.preprocessed_dir,
        task=args.task,
        split='val',
    )
    
    # Distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    ) if world_size > 1 else None
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    ) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    if is_main:
        print(f"  ✓ Train samples: {len(train_dataset)}")
        print(f"  ✓ Val samples: {len(val_dataset)}")
        print(f"  ✓ Effective batch size: {args.batch_size * world_size}")
    
    # Load diffusion models
    if is_main:
        print("\n🎨 Loading diffusion models...")
    
    vae, unet, _, _, noise_scheduler = load_diffusion_models(
        device=device,
        load_vae=True,
        load_unet=True,
        load_text_encoder=False,
    )
    
    # Create adapter
    if is_main:
        print("\n🧠 Creating EEG adapter...")
    
    if args.adapter_type == 'video':
        adapter = VideoLatentAdapter(
            in_channels=args.in_channels,
            eeg_time_steps=args.eeg_time_steps,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            use_glmnet=args.use_glmnet,
            num_subjects=args.num_subjects,
        ).to(device)
    else:
        adapter = EEGAdapter(
            in_channels=args.in_channels,
            eeg_time_steps=args.eeg_time_steps,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            use_glmnet=args.use_glmnet,
            num_subjects=args.num_subjects,
        ).to(device)
    
    if is_main:
        print(f"  ✓ Adapter parameters: {sum(p.numel() for p in adapter.parameters()) / 1e6:.2f}M")
    
    # Wrap with DDP
    if world_size > 1:
        adapter = DDP(adapter, device_ids=[local_rank], output_device=local_rank)
        if is_main:
            print(f"  ✓ DDP enabled across {world_size} GPUs")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    
    # Create trainer
    trainer = EEG2VideoTrainer(
        adapter=adapter,
        vae=vae,
        unet=unet,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        device=device,
        use_semantic_loss=args.use_semantic_loss,
        semantic_loss_weight=args.semantic_loss_weight,
        use_perceptual_loss=args.use_perceptual_loss,
        perceptual_loss_weight=args.perceptual_loss_weight,
        use_wandb=args.wandb,
        rank=rank,
    )
    
    # Load semantic embeddings
    if args.use_semantic_loss and args.clip_embeddings_path:
        if is_main:
            print(f"\n📝 Loading CLIP embeddings from: {args.clip_embeddings_path}")
        trainer.load_semantic_embeddings(Path(args.clip_embeddings_path))
    
    # Resume from checkpoint
    start_epoch = 1
    if args.resume:
        if is_main:
            print(f"\n♻️  Resuming from: {args.resume}")
        model_to_load = adapter.module if world_size > 1 else adapter
        ckpt = load_checkpoint(Path(args.resume), model_to_load, optimizer, device)
        start_epoch = ckpt.get('epoch', 0) + 1
        if is_main:
            print(f"  ✓ Resumed from epoch {start_epoch - 1}")
    
    # Training loop
    if is_main:
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        
        # Print metrics (only main process)
        if is_main:
            print(f"\nEpoch {epoch}/{args.epochs}")
            print_metrics(train_metrics, prefix="Train:")
            print_metrics(val_metrics, prefix="Val:")
        
        # Log to wandb
        if args.wandb and is_main:
            wandb.log({
                'epoch': epoch,
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()},
            })
        
        # Save checkpoint (only main process)
        if is_main:
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
            
            model_to_save = adapter.module if world_size > 1 else adapter
            save_checkpoint(
                output_dir / f"checkpoint_epoch{epoch:03d}.pth",
                model_to_save,
                optimizer,
                epoch,
                {'train': train_metrics, 'val': val_metrics},
            )
            
            if is_best:
                save_checkpoint(
                    output_dir / "best_model.pth",
                    model_to_save,
                    optimizer,
                    epoch,
                    {'train': train_metrics, 'val': val_metrics},
                )
                print(f"  💾 Saved best model (val_loss: {best_val_loss:.4f})")
            
            print()
    
    if is_main:
        print("=" * 70)
        print("✅ TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {output_dir}")
    
    if args.wandb and is_main:
        wandb.finish()
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
