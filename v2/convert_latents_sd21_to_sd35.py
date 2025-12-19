#!/usr/bin/env python3
"""
Convert SD 2.1 latents (4 channels) to SD 3.5 latents (16 channels)
This re-encodes video frames using SD 3.5's VAE
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# CRITICAL: Set cache before imports
import os
os.environ['HF_HOME'] = '/local-scratch/marios-datasets/hf-cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/local-scratch/marios-datasets/hf-cache/hub'

from diffusers import AutoencoderKL


def convert_latents(
    input_dir: Path,
    output_dir: Path,
    task: str,
    device: torch.device,
    batch_size: int = 8,
):
    """
    Convert SD 2.1 latents to SD 3.5 latents by re-encoding video frames
    
    Args:
        input_dir: Directory with SD 2.1 preprocessed data
        output_dir: Directory to save SD 3.5 preprocessed data
        task: Task name (dme, tp, inscapes)
        device: Device to use
        batch_size: Batch size for encoding
    """
    print("=" * 70)
    print("CONVERTING SD 2.1 LATENTS → SD 3.5 LATENTS")
    print("=" * 70)
    
    # Load video frames (these don't change)
    video_path = input_dir / f"{task}_video_frames.npy"
    if not video_path.exists():
        raise FileNotFoundError(f"Video frames not found: {video_path}")
    
    print(f"\n📂 Loading video frames (memory-mapped)...")
    video_frames = np.load(video_path, mmap_mode='r')
    num_segments, num_frames, H, W, C = video_frames.shape
    print(f"  ✓ Shape: {video_frames.shape}")
    print(f"  ✓ Segments: {num_segments:,}")
    print(f"  ✓ Frames per segment: {num_frames}")
    
    # Load SD 3.5 VAE
    print(f"\n🎨 Loading SD 3.5 VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(device)
    vae.eval()
    
    print(f"  ✓ VAE loaded")
    print(f"  ✓ Latent channels: {vae.config.latent_channels}")
    print(f"  ✓ Scaling factor: {vae.config.scaling_factor}")
    
    # Calculate output dimensions
    latent_H = H // 8
    latent_W = W // 8
    latent_C = vae.config.latent_channels  # Should be 16
    
    print(f"\n📐 Latent dimensions:")
    print(f"  Input video: [{H}, {W}, 3]")
    print(f"  Output latent: [{latent_C}, {latent_H}, {latent_W}]")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create memory-mapped output file for latents
    output_latents_path = output_dir / f"{task}_vae_latents_hd.npy"
    print(f"\n💾 Creating output file: {output_latents_path}")
    
    latents_mmap = np.lib.format.open_memmap(
        output_latents_path,
        mode='w+',
        dtype=np.float32,
        shape=(num_segments, num_frames, latent_C, latent_H, latent_W)
    )
    
    # Encode frames segment by segment
    print(f"\n🔄 Encoding {num_segments:,} segments...")
    
    with torch.no_grad():
        for seg_idx in tqdm(range(num_segments), desc="Encoding"):
            # Load segment frames
            segment_frames = video_frames[seg_idx]  # [num_frames, H, W, 3]
            
            # Process frames in batches
            for frame_start in range(0, num_frames, batch_size):
                frame_end = min(frame_start + batch_size, num_frames)
                batch_frames = segment_frames[frame_start:frame_end]
                
                # Convert to tensor [B, 3, H, W] in [-1, 1]
                # .copy() makes it writable and suppresses the warning
                frames_tensor = torch.from_numpy(batch_frames.copy()).float() / 127.5 - 1.0
                frames_tensor = frames_tensor.permute(0, 3, 1, 2).to(device)
                
                # Encode with VAE
                latents = vae.encode(frames_tensor).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Store in memory-mapped file
                latents_np = latents.cpu().numpy()
                latents_mmap[seg_idx, frame_start:frame_end] = latents_np
            
            # Flush periodically
            if seg_idx % 100 == 0:
                latents_mmap.flush()
    
    # Final flush
    latents_mmap.flush()
    
    print(f"\n✓ Encoded latents saved to: {output_latents_path}")
    print(f"  Shape: {latents_mmap.shape}")
    print(f"  Size: {output_latents_path.stat().st_size / 1e9:.2f} GB")
    
    # Copy other files that don't need conversion
    print(f"\n📋 Copying additional files...")
    
    files_to_copy = [
        f"{task}_captions_hd.json",
        f"{task}_flow_scores_hd.npy",
        f"{task}_clip_text_embeddings.npy",
        f"{task}_eeg.npy",
    ]
    
    for filename in files_to_copy:
        src = input_dir / filename
        dst = output_dir / filename
        
        if src.exists():
            print(f"  Copying: {filename}")
            if filename.endswith('.json'):
                import shutil
                shutil.copy2(src, dst)
            else:
                # For numpy files, use memory mapping to avoid loading into RAM
                data = np.load(src, mmap_mode='r')
                np.save(dst, data)
        else:
            print(f"  ⚠️  Not found: {filename}")
    
    # Save conversion metadata
    metadata = {
        'source_model': 'stabilityai/stable-diffusion-2-1',
        'target_model': 'stabilityai/stable-diffusion-3.5-medium',
        'source_latent_channels': 4,
        'target_latent_channels': latent_C,
        'num_segments': int(num_segments),
        'num_frames_per_segment': int(num_frames),
        'video_resolution': [int(H), int(W)],
        'latent_resolution': [int(latent_H), int(latent_W)],
    }
    
    metadata_path = output_dir / f"{task}_conversion_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Saved metadata to: {metadata_path}")
    
    print("\n" + "=" * 70)
    print("✅ CONVERSION COMPLETE!")
    print("=" * 70)
    print(f"Old latents (4ch): {input_dir}")
    print(f"New latents (16ch): {output_dir}")
    print(f"\nYou can now train with SD 3.5 using:")
    print(f"  --preprocessed-dir {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SD 2.1 latents to SD 3.5 latents"
    )
    
    parser.add_argument('--input-dir', required=True,
                       help='Directory with SD 2.1 preprocessed data')
    parser.add_argument('--output-dir', required=True,
                       help='Directory to save SD 3.5 preprocessed data')
    parser.add_argument('--task', required=True,
                       choices=['dme', 'tp', 'inscapes'],
                       help='Task name')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for VAE encoding')
    parser.add_argument('--device', default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"❌ Error: Input directory does not exist: {input_dir}")
        return
    
    convert_latents(
        input_dir=input_dir,
        output_dir=output_dir,
        task=args.task,
        device=device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
