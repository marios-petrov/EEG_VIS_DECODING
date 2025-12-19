#!/usr/bin/env python3
"""
Preprocess SEED-DV dataset for EEGMamba + SD 3.5 training

Matches EEG2Video paper setup (NeurIPS 2024):
- EEG: 62 channels, 200 Hz, 2-second segments
- Video: 6 frames per segment (3 FPS), resized to 512x288 (or 768x768 for SD 3.5)
- Train: Blocks 1-6, Test: Block 7

SEED-DV structure:
- EEG shape: (7 blocks, 62 channels, 104000 samples) per subject @ 200Hz
- Each block: 520 seconds = 40 concepts × (3s hint + 5×2s clips)
- Videos: 7 × 10min @ 24fps, 1920×1080

Usage:
    python preprocess_seed_dv.py \
        --seed-dir /local-scratch/SEED \
        --output-dir /local-scratch/marios-datasets/SEED/preprocessed_sd35 \
        --target-size 768

For SD 3.5, use --target-size 768
For comparison with EEG2Video (SD 1.4), use --target-size 512
"""

import os
os.environ['HF_HOME'] = '/local-scratch/marios-datasets/hf-cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/local-scratch/marios-datasets/hf-cache/hub'

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SegmentInfo:
    """Information about a single EEG-video segment"""
    subject_id: int
    block_id: int
    concept_id: int
    clip_id: int  # 0-4 within concept
    segment_idx: int  # global index
    eeg_start: int
    eeg_end: int
    video_start_frame: int
    video_end_frame: int
    caption: str


def load_subject_eeg(eeg_dir: Path, subject_id: int) -> Optional[np.ndarray]:
    """Load EEG data for a subject"""
    eeg_path = eeg_dir / f"sub{subject_id}.npy"
    if eeg_path.exists():
        data = np.load(eeg_path, allow_pickle=True)
        return data
    return None


def load_blip_captions(caption_dir: Path) -> Dict[int, List[str]]:
    """Load BLIP captions for all 7 video blocks"""
    captions = {}
    
    for block_idx in range(7):
        block_num = block_idx + 1
        # Handle ordinal suffixes
        if block_num == 1:
            suffix = "1st"
        elif block_num == 2:
            suffix = "2nd"
        elif block_num == 3:
            suffix = "3rd"
        else:
            suffix = f"{block_num}th"
        
        caption_path = caption_dir / f"{suffix}_10min.txt"
        
        if caption_path.exists():
            with open(caption_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            captions[block_idx] = lines
            print(f"  Block {block_num}: {len(lines)} captions")
        else:
            print(f"  Warning: {caption_path} not found")
            captions[block_idx] = ["a video frame"] * 200  # 40 concepts × 5 clips
    
    return captions


def extract_video_frames_for_block(
    video_path: Path,
    target_fps: float = 3.0,
    target_size: Tuple[int, int] = (768, 768),
) -> np.ndarray:
    """
    Extract and preprocess frames from a video block.
    
    Each block is ~520 seconds at 24fps = 12480 frames
    At 3 FPS: 1560 frames per block
    """
    cap = cv2.VideoCapture(str(video_path))
    
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"    Video: {orig_fps:.1f} fps, {total_frames} frames, {total_frames/orig_fps:.1f}s")
    
    # Calculate frame sampling
    frame_interval = orig_fps / target_fps  # e.g., 24/3 = 8
    
    frames = []
    frame_idx = 0
    next_sample = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx >= next_sample:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
            next_sample += frame_interval
        
        frame_idx += 1
    
    cap.release()
    print(f"    Extracted {len(frames)} frames at {target_fps} FPS")
    return np.array(frames)


def compute_segment_timing(
    block_idx: int,
    concept_idx: int,
    clip_idx: int,
    eeg_fs: int = 200,
    video_fps: float = 3.0,
) -> Tuple[int, int, int, int]:
    """
    Compute EEG and video timing for a specific clip.
    
    Block structure (per concept):
    - 3 second hint ("Next you will see: XX")
    - 5 × 2-second video clips
    Total per concept: 3 + 10 = 13 seconds
    
    For 40 concepts: 40 × 13 = 520 seconds per block
    """
    # Time offset for this concept within the block
    concept_start_sec = concept_idx * 13  # 13 seconds per concept
    
    # Skip the 3-second hint, then find the specific clip
    clip_start_sec = concept_start_sec + 3 + (clip_idx * 2)  # 2 seconds per clip
    clip_end_sec = clip_start_sec + 2
    
    # EEG timing (200 Hz)
    eeg_start = int(clip_start_sec * eeg_fs)
    eeg_end = int(clip_end_sec * eeg_fs)
    
    # Video timing (3 FPS for our downsampled video)
    video_start = int(clip_start_sec * video_fps)
    video_end = int(clip_end_sec * video_fps)
    
    return eeg_start, eeg_end, video_start, video_end


def encode_frames_to_latents(
    frames: np.ndarray,
    vae,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    """Encode video frames to SD 3.5 VAE latents"""
    from torchvision import transforms
    
    h, w = frames.shape[1:3]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    latents = []
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        
        # Transform frames
        batch_tensors = torch.stack([transform(f) for f in batch_frames]).to(device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                latent = vae.encode(batch_tensors).latent_dist.sample()
                latent = latent * vae.config.scaling_factor
        
        latents.append(latent.cpu().numpy())
    
    return np.concatenate(latents, axis=0)


def main():
    parser = argparse.ArgumentParser(description='Preprocess SEED-DV for EEGMamba + SD 3.5')
    parser.add_argument('--seed-dir', type=str, default='/local-scratch/SEED',
                       help='Path to SEED-DV dataset')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for preprocessed data')
    parser.add_argument('--target-size', type=int, default=768,
                       help='Target frame size (768 for SD 3.5, 512 for SD 1.4)')
    parser.add_argument('--target-fps', type=float, default=3.0,
                       help='Target FPS for video (3.0 matches EEG2Video paper)')
    parser.add_argument('--frames-per-segment', type=int, default=6,
                       help='Frames per 2-second segment (6 = 3 FPS)')
    parser.add_argument('--subjects', type=str, default='1-20',
                       help='Subject range to process (e.g., "1-20" or "1,2,5")')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip-latent-encoding', action='store_true',
                       help='Skip VAE encoding (for testing pipeline)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip subjects that already have output files')
    args = parser.parse_args()
    
    seed_dir = Path(args.seed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Parse subject range
    if '-' in args.subjects:
        start, end = map(int, args.subjects.split('-'))
        subject_ids = list(range(start, end + 1))
    else:
        subject_ids = [int(s) for s in args.subjects.split(',')]
    
    print(f"Processing subjects: {subject_ids}")
    print(f"Target size: {args.target_size}x{args.target_size}")
    print(f"Target FPS: {args.target_fps}")
    
    # Load BLIP captions
    print("\n📝 Loading BLIP captions...")
    caption_dir = seed_dir / "Video" / "BLIP-caption"
    captions = load_blip_captions(caption_dir)
    
    # Load VAE for latent encoding
    vae = None
    if not args.skip_latent_encoding:
        print("\n🔧 Loading SD 3.5 VAE...")
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            subfolder="vae",
            torch_dtype=torch.float16,
        ).to(device)
        vae.eval()
    
    # Video paths
    video_dir = seed_dir / "Video"
    video_files = {
        0: "1st_10min.mp4",
        1: "2nd_10min.mp4",
        2: "3rd_10min.mp4",
        3: "4th_10min.mp4",
        4: "5th_10min.mp4",
        5: "6th_10min.mp4",
        6: "7th_10min.mp4",
    }
    
    # Process each subject
    eeg_dir = seed_dir / "EEG"
    
    for subj_id in subject_ids:
        print(f"\n{'='*60}")
        print(f"Processing Subject {subj_id}")
        print('='*60)
        
        # Check if already processed
        subj_output = output_dir / f"sub{subj_id}"
        if args.skip_existing and (subj_output / "metadata.json").exists():
            print(f"  Skipping (already processed)")
            continue
        
        # Load EEG
        eeg_data = load_subject_eeg(eeg_dir, subj_id)
        if eeg_data is None:
            print(f"  EEG file not found, skipping")
            continue
        
        print(f"  EEG shape: {eeg_data.shape}")  # Should be (7, 62, 104000)
        
        if eeg_data.shape != (7, 62, 104000):
            print(f"  Warning: Unexpected EEG shape, expected (7, 62, 104000)")
        
        subj_output.mkdir(parents=True, exist_ok=True)
        
        all_segments = []
        all_eeg = []
        all_latents = []  # Will be populated if VAE encoding is done
        all_frames = []  # Store frames for VAE encoding
        
        # Process each block
        for block_idx in range(7):
            print(f"\n  📦 Block {block_idx + 1}/7")
            
            # Extract video frames for this block
            video_path = video_dir / video_files[block_idx]
            if not video_path.exists():
                print(f"    Video not found: {video_path}")
                continue
            
            block_frames = extract_video_frames_for_block(
                video_path,
                target_fps=args.target_fps,
                target_size=(args.target_size, args.target_size),
            )
            
            # Get EEG for this block
            block_eeg = eeg_data[block_idx]  # (62, 104000)
            
            # Get captions for this block
            block_captions = captions.get(block_idx, ["a video frame"] * 200)
            
            # Process each concept (40 per block)
            caption_idx = 0
            for concept_idx in range(40):
                # Process each clip within the concept (5 clips)
                for clip_idx in range(5):
                    # Get timing
                    eeg_start, eeg_end, vid_start, vid_end = compute_segment_timing(
                        block_idx, concept_idx, clip_idx,
                        eeg_fs=200, video_fps=args.target_fps
                    )
                    
                    # Extract EEG segment (2 seconds = 400 samples at 200Hz)
                    if eeg_end <= block_eeg.shape[1]:
                        eeg_segment = block_eeg[:, eeg_start:eeg_end]
                    else:
                        print(f"    Warning: EEG index out of bounds for concept {concept_idx}, clip {clip_idx}")
                        continue
                    
                    # Extract video frames (6 frames at 3 FPS)
                    if vid_end <= len(block_frames):
                        video_segment = block_frames[vid_start:vid_end]
                    else:
                        print(f"    Warning: Video index out of bounds for concept {concept_idx}, clip {clip_idx}")
                        continue
                    
                    # Get caption
                    caption = block_captions[caption_idx] if caption_idx < len(block_captions) else "a video frame"
                    caption_idx += 1
                    
                    # Create segment info
                    seg_info = {
                        'subject_id': subj_id,
                        'block_id': block_idx,
                        'concept_id': concept_idx,
                        'clip_id': clip_idx,
                        'segment_idx': len(all_segments),
                        'caption': caption,
                        'is_train': block_idx < 6,  # Blocks 0-5 train, block 6 test
                    }
                    
                    all_segments.append(seg_info)
                    all_eeg.append(eeg_segment)
                    all_frames.append(video_segment)
            
            print(f"    Processed {len(all_segments)} segments so far")
        
        # Convert to arrays
        all_eeg = np.array(all_eeg)  # (N, 62, 400)
        all_frames = np.array(all_frames)  # (N, 6, H, W, 3)
        
        print(f"\n  📊 Subject {subj_id} totals:")
        print(f"    EEG: {all_eeg.shape}")
        print(f"    Frames: {all_frames.shape}")
        print(f"    Segments: {len(all_segments)}")
        
        # Encode frames to latents
        if vae is not None:
            print(f"\n  🔄 Encoding frames to latents...")
            # Flatten frames for batch encoding
            n_segments, n_frames = all_frames.shape[:2]
            flat_frames = all_frames.reshape(-1, *all_frames.shape[2:])
            
            flat_latents = encode_frames_to_latents(flat_frames, vae, device, batch_size=16)
            
            # Reshape back to (N, frames, C, H, W)
            latent_shape = flat_latents.shape[1:]
            all_latents = flat_latents.reshape(n_segments, n_frames, *latent_shape)
            
            print(f"    Latents: {all_latents.shape}")
            
            # Save latents
            np.save(subj_output / "latents.npy", all_latents.astype(np.float16))
        
        # Save EEG
        np.save(subj_output / "eeg.npy", all_eeg.astype(np.float32))
        
        # Save frames (optional, for visualization)
        # np.save(subj_output / "frames.npy", all_frames.astype(np.uint8))
        
        # Save metadata
        metadata = {
            'subject_id': subj_id,
            'n_segments': len(all_segments),
            'n_train': sum(1 for s in all_segments if s['is_train']),
            'n_test': sum(1 for s in all_segments if not s['is_train']),
            'eeg_shape': list(all_eeg.shape),
            'frames_shape': list(all_frames.shape),
            'latents_shape': list(all_latents.shape) if len(all_latents) > 0 else None,
            'eeg_fs': 200,
            'video_fps': args.target_fps,
            'target_size': args.target_size,
            'segments': all_segments,
        }
        
        with open(subj_output / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n  ✅ Saved to {subj_output}")
        print(f"    Train segments: {metadata['n_train']}")
        print(f"    Test segments: {metadata['n_test']}")
    
    # Create global index
    print("\n📋 Creating global index...")
    global_index = {
        'subjects': subject_ids,
        'n_concepts': 40,
        'n_clips_per_concept': 5,
        'n_blocks': 7,
        'train_blocks': [0, 1, 2, 3, 4, 5],
        'test_blocks': [6],
        'eeg_channels': 62,
        'eeg_fs': 200,
        'segment_duration': 2.0,
        'frames_per_segment': args.frames_per_segment,
        'target_size': args.target_size,
    }
    
    with open(output_dir / "index.json", 'w') as f:
        json.dump(global_index, f, indent=2)
    
    print(f"\n✅ Preprocessing complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
