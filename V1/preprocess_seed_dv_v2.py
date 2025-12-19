#!/usr/bin/env python3
"""
Preprocess SEED-DV dataset for EEGMamba + SD 3.5 training (CORRECTED VERSION)

KEY FIX: Uses All_video_label.npy to get actual concept IDs per block!
- SEED-DV concept IDs are 1-40 (not 0-39)
- Each block has concepts in DIFFERENT randomized order
- Must use All_video_label.npy to map position → concept_id

SEED-DV Concept Mapping (from paper Figure 1):
  1-7:   Land Animal (Cat, Dog, Elephant, Horse, Panda, Rabbit, Bird)
  8-11:  Water Animal (Fish, Jellyfish, Shark, Turtle)
  12-14: Plant (Flower, Mushroom, Tree)
  15-18: Exercise (Boxing, Dancing, Running, Skiing)
  19-21: Human (Couple, Face, Crowd)
  22-27: Natural Scene (Beach, Buildings, Mountain, Road, Water, Fireworks)
  28-32: Food (Banana, Cake, Drink, Pizza, Watermelon)
  33-35: Musical (Drum, Guitar, Piano)
  36-40: Transportation (Bike, Car, Hot balloon, Airplane, Ship)

Animals = 1-11, Non-animals = 12-40

Usage:
    python preprocess_seed_dv_v2.py \
        --seed-dir /local-scratch/SEED \
        --output-dir /local-scratch/marios-datasets/SEED/preprocessed_sd35_v2 \
        --target-size 768
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

# ============================================================================
# SEED-DV CONCEPT DEFINITIONS (1-indexed as in paper)
# ============================================================================
SEED_DV_CONCEPTS = {
    1: "cat", 2: "dog", 3: "elephant", 4: "horse", 5: "panda", 
    6: "rabbit", 7: "bird", 8: "fish", 9: "jellyfish", 10: "shark",
    11: "turtle", 12: "flower", 13: "mushroom", 14: "tree", 15: "boxing",
    16: "dancing", 17: "running", 18: "skiing", 19: "couple", 20: "face",
    21: "crowd", 22: "beach", 23: "buildings", 24: "mountain", 25: "road",
    26: "water", 27: "fireworks", 28: "banana", 29: "cake", 30: "drink",
    31: "pizza", 32: "watermelon", 33: "drum", 34: "guitar", 35: "piano",
    36: "bike", 37: "car", 38: "hot air balloon", 39: "airplane", 40: "ship",
}

CONCEPT_COARSE = {
    **{i: "land_animal" for i in range(1, 8)},
    **{i: "water_animal" for i in range(8, 12)},
    **{i: "plant" for i in range(12, 15)},
    **{i: "exercise" for i in range(15, 19)},
    **{i: "human" for i in range(19, 22)},
    **{i: "natural_scene" for i in range(22, 28)},
    **{i: "food" for i in range(28, 33)},
    **{i: "musical" for i in range(33, 36)},
    **{i: "transportation" for i in range(36, 41)},
}

ANIMAL_CONCEPTS = set(range(1, 12))  # 1-11 are animals


def load_video_metadata(meta_dir: Path) -> Dict:
    """Load all video metadata files"""
    metadata = {}
    
    # Concept labels per block: shape (7, 40) - order of concepts in each block
    label_path = meta_dir / "All_video_label.npy"
    if label_path.exists():
        metadata['labels'] = np.load(label_path)  # (7, 40)
        print(f"  Labels shape: {metadata['labels'].shape}")
        print(f"  Block 0 concept order (first 10): {metadata['labels'][0, :10]}")
    else:
        print(f"  WARNING: {label_path} not found! Using sequential order.")
        metadata['labels'] = np.tile(np.arange(1, 41), (7, 1))
    
    # Optional metadata
    for name in ['color', 'human_apperance', 'face_apperance', 'obj_number', 'optical_flow_score']:
        path = meta_dir / f"All_video_{name}.npy"
        if path.exists():
            metadata[name] = np.load(path)
            print(f"  {name} shape: {metadata[name].shape}")
    
    return metadata


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
            captions[block_idx] = ["a video frame"] * 200
    
    return captions


def extract_video_frames_for_block(
    video_path: Path,
    target_fps: float = 3.0,
    target_size: Tuple[int, int] = (768, 768),
) -> np.ndarray:
    """Extract and preprocess frames from a video block"""
    cap = cv2.VideoCapture(str(video_path))
    
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"    Video: {orig_fps:.1f} fps, {total_frames} frames, {total_frames/orig_fps:.1f}s")
    
    frame_interval = orig_fps / target_fps
    
    frames = []
    frame_idx = 0
    next_sample = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx >= next_sample:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
            next_sample += frame_interval
        
        frame_idx += 1
    
    cap.release()
    print(f"    Extracted {len(frames)} frames at {target_fps} FPS")
    return np.array(frames)


def compute_segment_timing(
    concept_position: int,  # Position within block (0-39)
    clip_idx: int,          # Clip within concept (0-4)
    eeg_fs: int = 200,
    video_fps: float = 3.0,
) -> Tuple[int, int, int, int]:
    """
    Compute EEG and video timing for a specific clip.
    
    Block structure per concept:
    - 3 second hint
    - 5 × 2-second video clips
    Total per concept: 13 seconds
    """
    concept_start_sec = concept_position * 13
    clip_start_sec = concept_start_sec + 3 + (clip_idx * 2)
    clip_end_sec = clip_start_sec + 2
    
    eeg_start = int(clip_start_sec * eeg_fs)
    eeg_end = int(clip_end_sec * eeg_fs)
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
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    latents = []
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        batch_tensors = torch.stack([transform(f) for f in batch_frames]).to(device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                latent = vae.encode(batch_tensors).latent_dist.sample()
                latent = latent * vae.config.scaling_factor
        
        latents.append(latent.cpu().numpy())
    
    return np.concatenate(latents, axis=0)


def main():
    parser = argparse.ArgumentParser(description='Preprocess SEED-DV (corrected version)')
    parser.add_argument('--seed-dir', type=str, default='/local-scratch/SEED')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--target-size', type=int, default=768)
    parser.add_argument('--target-fps', type=float, default=3.0)
    parser.add_argument('--frames-per-segment', type=int, default=6)
    parser.add_argument('--subjects', type=str, default='1-20')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip-latent-encoding', action='store_true')
    parser.add_argument('--skip-existing', action='store_true')
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
    
    # Load video metadata (CRITICAL: concept labels per block!)
    print("\n📊 Loading video metadata...")
    meta_dir = seed_dir / "Video" / "meta-info"
    video_meta = load_video_metadata(meta_dir)
    block_labels = video_meta['labels']  # (7, 40) - concept IDs per block position
    
    # Load BLIP captions
    print("\n📝 Loading BLIP captions...")
    caption_dir = seed_dir / "Video" / "BLIP-caption"
    captions = load_blip_captions(caption_dir)
    
    # Load VAE
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
    video_files = {i: f"{['1st','2nd','3rd','4th','5th','6th','7th'][i]}_10min.mp4" for i in range(7)}
    
    # EEG directory
    eeg_dir = seed_dir / "EEG"
    
    for subj_id in subject_ids:
        print(f"\n{'='*60}")
        print(f"Processing Subject {subj_id}")
        print('='*60)
        
        subj_output = output_dir / f"sub{subj_id}"
        if args.skip_existing and (subj_output / "metadata.json").exists():
            print(f"  Skipping (already processed)")
            continue
        
        # Load EEG
        eeg_data = load_subject_eeg(eeg_dir, subj_id)
        if eeg_data is None:
            print(f"  EEG file not found, skipping")
            continue
        
        print(f"  EEG shape: {eeg_data.shape}")
        
        subj_output.mkdir(parents=True, exist_ok=True)
        
        all_segments = []
        all_eeg = []
        all_frames = []
        
        # Process each block
        for block_idx in range(7):
            print(f"\n  📦 Block {block_idx + 1}/7")
            
            # Get concept order for this block
            concept_order = block_labels[block_idx]  # Array of 40 concept IDs (1-40)
            print(f"    Concept order (first 5): {concept_order[:5]} → {[SEED_DV_CONCEPTS[c] for c in concept_order[:5]]}")
            
            # Extract video frames
            video_path = video_dir / video_files[block_idx]
            if not video_path.exists():
                print(f"    Video not found: {video_path}")
                continue
            
            block_frames = extract_video_frames_for_block(
                video_path,
                target_fps=args.target_fps,
                target_size=(args.target_size, args.target_size),
            )
            
            block_eeg = eeg_data[block_idx]  # (62, 104000)
            block_captions = captions.get(block_idx, ["a video frame"] * 200)
            
            # Process each concept position (40 per block)
            caption_idx = 0
            for position in range(40):
                # Get ACTUAL concept ID from metadata (1-indexed!)
                concept_id = int(concept_order[position])
                concept_name = SEED_DV_CONCEPTS.get(concept_id, "unknown")
                coarse_class = CONCEPT_COARSE.get(concept_id, "unknown")
                is_animal = concept_id in ANIMAL_CONCEPTS
                
                # Process each clip (5 per concept)
                for clip_idx in range(5):
                    eeg_start, eeg_end, vid_start, vid_end = compute_segment_timing(
                        position, clip_idx,
                        eeg_fs=200, video_fps=args.target_fps
                    )
                    
                    # Extract EEG segment
                    if eeg_end <= block_eeg.shape[1]:
                        eeg_segment = block_eeg[:, eeg_start:eeg_end]
                    else:
                        continue
                    
                    # Extract video frames
                    if vid_end <= len(block_frames):
                        video_segment = block_frames[vid_start:vid_end]
                    else:
                        continue
                    
                    # Get caption
                    caption = block_captions[caption_idx] if caption_idx < len(block_captions) else f"a video of {concept_name}"
                    caption_idx += 1
                    
                    # Create segment info with CORRECT concept_id
                    seg_info = {
                        'subject_id': subj_id,
                        'block_id': block_idx,
                        'concept_id': concept_id,          # 1-indexed (1-40)
                        'concept_name': concept_name,
                        'coarse_class': coarse_class,
                        'is_animal': is_animal,
                        'position_in_block': position,      # 0-39
                        'clip_id': clip_idx,
                        'segment_idx': len(all_segments),
                        'caption': caption,
                        'is_train': block_idx < 6,
                    }
                    
                    all_segments.append(seg_info)
                    all_eeg.append(eeg_segment)
                    all_frames.append(video_segment)
            
            print(f"    Processed {len(all_segments)} segments so far")
        
        # Convert to arrays
        all_eeg = np.array(all_eeg)
        all_frames = np.array(all_frames)
        
        print(f"\n  📊 Subject {subj_id} totals:")
        print(f"    EEG: {all_eeg.shape}")
        print(f"    Frames: {all_frames.shape}")
        print(f"    Segments: {len(all_segments)}")
        
        # Verify concept distribution
        concept_counts = {}
        for seg in all_segments:
            cid = seg['concept_id']
            concept_counts[cid] = concept_counts.get(cid, 0) + 1
        print(f"    Concepts represented: {len(concept_counts)}")
        
        # Encode frames to latents
        all_latents = []
        if vae is not None:
            print(f"\n  🔄 Encoding frames to latents...")
            n_segments, n_frames = all_frames.shape[:2]
            flat_frames = all_frames.reshape(-1, *all_frames.shape[2:])
            flat_latents = encode_frames_to_latents(flat_frames, vae, device, batch_size=16)
            latent_shape = flat_latents.shape[1:]
            all_latents = flat_latents.reshape(n_segments, n_frames, *latent_shape)
            print(f"    Latents: {all_latents.shape}")
            np.save(subj_output / "latents.npy", all_latents.astype(np.float16))
        
        # Save EEG
        np.save(subj_output / "eeg.npy", all_eeg.astype(np.float32))
        
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
            'concept_id_range': '1-40 (1-indexed)',
            'animal_concepts': list(ANIMAL_CONCEPTS),
            'segments': all_segments,
        }
        
        with open(subj_output / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n  ✅ Saved to {subj_output}")
    
    # Create global index with concept definitions
    print("\n📋 Creating global index...")
    global_index = {
        'subjects': subject_ids,
        'n_concepts': 40,
        'concept_id_range': '1-40 (1-indexed)',
        'concepts': SEED_DV_CONCEPTS,
        'coarse_classes': list(set(CONCEPT_COARSE.values())),
        'animal_concepts': list(ANIMAL_CONCEPTS),
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
