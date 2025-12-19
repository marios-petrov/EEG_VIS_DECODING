#!/usr/bin/env python3
"""
Fine-tune Stable Diffusion UNet on preprocessed video latents
Optimized for 8x V100 32GB GPUs with DistributedDataParallel
"""

import argparse
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from models import load_diffusion_models, EMA
from utils import LatentFramesWithPrompts, seed_everything


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
    parser = argparse.ArgumentParser(description="Fine-tune SD 2.1 UNet (Multi-GPU)")
    
    # Data arguments
    parser.add_argument('--task', required=True, choices=['dme', 'tp', 'inscapes'])
    parser.add_argument('--preprocessed-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16, help='Per-GPU batch size')
    parser.add_argument('--grad-accum', type=int, default=1, help='Gradient accumulation')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    
    # Optimization (only performance-beneficial ones)
    parser.add_argument('--precision', default='fp16', choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument('--xformers', action='store_true', default=True,
                       help='Use xFormers (faster, recommended)')
    parser.add_argument('--no-xformers', dest='xformers', action='store_false')
    
    # System arguments
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    
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
        print("=" * 70)
        print(f"FINE-TUNING UNET - {world_size} GPUs")
        print("=" * 70)
    
    # Load dataset
    if is_main:
        print(f"\n📊 Loading dataset...")
    
    dataset = LatentFramesWithPrompts(
        preprocessed_dir=Path(args.preprocessed_dir),
        task=args.task,
    )
    
    # Distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    ) if world_size > 1 else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    if is_main:
        print(f"  ✓ Dataset size: {len(dataset)}")
        print(f"  ✓ Batches per GPU: {len(dataloader)}")
        print(f"  ✓ Effective batch size: {args.batch_size * world_size * args.grad_accum}")
    
    # Load models (only on rank 0, then broadcast)
    if is_main:
        print(f"\n🎨 Loading Stable Diffusion models...")
    
    _, unet, text_encoder, tokenizer, noise_scheduler = load_diffusion_models(
        model_name="stabilityai/stable-diffusion-2-1",
        device=device,
        load_vae=False,
        load_unet=True,
        load_text_encoder=True,
    )
    
    # Freeze text encoder
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False
    
    # Apply xformers if available (KEEP - it's faster, not just memory saving)
    if args.xformers:
        try:
            from diffusers.utils.import_utils import is_xformers_available
            if is_xformers_available():
                unet.enable_xformers_memory_efficient_attention()
                if is_main:
                    print("  ✓ xFormers enabled (faster attention)")
        except Exception as e:
            if is_main:
                print(f"  ⚠️  xFormers failed: {e}")
    
    # Wrap with DDP
    if world_size > 1:
        unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)
        if is_main:
            print(f"  ✓ DDP enabled across {world_size} GPUs")
    
    unet.train()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    
    # EMA (only on main process)
    ema = EMA(unet.module if world_size > 1 else unet, mu=0.999, device='cpu') if is_main else None
    
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
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Number of GPUs: {world_size}")
        print(f"Gradient accumulation: {args.grad_accum}")
        print(f"Effective batch size: {args.batch_size * world_size * args.grad_accum}")
        print()
    
    global_step = 0
    
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        unet.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:03d}", disable=not is_main)
        
        for i, (latents, prompts) in enumerate(pbar):
            # Move latents to device
            latents = latents.to(device, non_blocking=True)
            
            # Encode text
            text_inputs = tokenizer(
                list(prompts),
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            with torch.no_grad():
                text_inputs = {k: v.to(device, non_blocking=True) for k, v in text_inputs.items()}
                text_embeddings = text_encoder(**text_inputs).last_hidden_state
            
            # Prepare noisy latents
            batch_size = latents.shape[0]
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Forward pass with mixed precision
            with torch.autocast(
                device_type='cuda',
                dtype=amp_dtype,
                enabled=(args.precision != 'fp32'),
            ):
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings,
                ).sample
                
                loss = F.mse_loss(noise_pred, noise)
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
                
                if ema is not None:
                    ema.update(unet.module if world_size > 1 else unet)
                
                global_step += 1
            
            # Update metrics
            epoch_loss += loss.item() * args.grad_accum
            
            if is_main:
                pbar.set_postfix(loss=f"{loss.item() * args.grad_accum:.4f}")
        
        # Epoch metrics
        if is_main:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")
        
        # Save checkpoint (only on main process)
        if is_main:
            print(f"  💾 Saving checkpoint...")
            
            # Store EMA weights
            model_to_save = unet.module if world_size > 1 else unet
            ema.store(model_to_save)
            
            checkpoint = {
                'unet': model_to_save.state_dict(),
                'noise_scheduler': noise_scheduler.config,
                'epoch': epoch,
                'global_step': global_step,
                'loss': avg_loss if is_main else 0,
            }
            
            torch.save(
                checkpoint,
                output_dir / f"unet_ema_ep{epoch:03d}.pth"
            )
            
            # Restore original weights
            ema.restore(model_to_save)
            
            print()
    
    if is_main:
        print("=" * 70)
        print("✅ FINE-TUNING COMPLETE!")
        print("=" * 70)
        print(f"Checkpoints saved to: {output_dir}")
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
