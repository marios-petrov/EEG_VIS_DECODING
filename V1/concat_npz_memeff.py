#!/usr/bin/env python3
"""
Memory-efficient concatenation of preprocessed .npz files
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm


def concatenate_with_memmap(
    npz_files: list,
    output_dir: Path,
    task_name: str,
):
    """
    Memory-efficient concatenation using memory-mapped files
    """
    print(f"\n{'='*70}")
    print("CONCATENATING DATA (Memory-Efficient)")
    print(f"{'='*70}")
    
    print(f"Found {len(npz_files)} files to concatenate")
    
    # First pass: get shapes
    print("\n📏 Scanning file dimensions...")
    total_segments = 0
    first_file = True
    eeg_shape = video_shape = latents_shape = None
    
    for npz_file in tqdm(npz_files, desc="Scanning"):
        print(f"  Loading {npz_file.name}...")
        with np.load(npz_file, allow_pickle=True) as data:
            total_segments += data['eeg'].shape[0]
            if first_file:
                eeg_shape = data['eeg'].shape[1:]
                video_shape = data['video'].shape[1:]
                latents_shape = data['latents'].shape[1:]
                first_file = False
    
    print(f"\n  Total segments: {total_segments}")
    print(f"  EEG shape per segment: {eeg_shape}")
    print(f"  Video shape per segment: {video_shape}")
    print(f"  Latents shape per segment: {latents_shape}")
    
    # Create memory-mapped output files
    eeg_out = output_dir / f"{task_name}_eeg.npy"
    video_out = output_dir / f"{task_name}_video_frames.npy"
    latents_out = output_dir / f"{task_name}_vae_latents_hd.npy"
    
    print(f"\n📝 Creating memory-mapped output files...")
    eeg_mmap = np.lib.format.open_memmap(
        eeg_out, mode='w+', dtype=np.float32,
        shape=(total_segments,) + eeg_shape
    )
    video_mmap = np.lib.format.open_memmap(
        video_out, mode='w+', dtype=np.uint8,
        shape=(total_segments,) + video_shape
    )
    latents_mmap = np.lib.format.open_memmap(
        latents_out, mode='w+', dtype=np.float32,
        shape=(total_segments,) + latents_shape
    )
    
    # Second pass: copy data in chunks
    print(f"\n📋 Copying data to memory-mapped files...")
    current_idx = 0
    
    for npz_file in tqdm(npz_files, desc="Copying"):
        with np.load(npz_file, allow_pickle=True) as data:
            n_segments = data['eeg'].shape[0]
            end_idx = current_idx + n_segments
            
            # Copy in chunks to avoid loading everything at once
            eeg_mmap[current_idx:end_idx] = data['eeg']
            video_mmap[current_idx:end_idx] = data['video']
            latents_mmap[current_idx:end_idx] = data['latents']
            
            current_idx = end_idx
            
            # Flush to disk periodically
            if current_idx % 1000 == 0:
                eeg_mmap.flush()
                video_mmap.flush()
                latents_mmap.flush()
    
    # Final flush
    print("\n💾 Flushing data to disk...")
    eeg_mmap.flush()
    video_mmap.flush()
    latents_mmap.flush()
    
    print(f"\n✅ Successfully created concatenated files:")
    print(f"  📊 EEG: {eeg_out}")
    print(f"     Shape: {eeg_mmap.shape}, Size: {eeg_out.stat().st_size / 1e9:.2f} GB")
    print(f"  🎬 Video: {video_out}")
    print(f"     Shape: {video_mmap.shape}, Size: {video_out.stat().st_size / 1e9:.2f} GB")
    print(f"  🔄 Latents: {latents_out}")
    print(f"     Shape: {latents_mmap.shape}, Size: {latents_out.stat().st_size / 1e9:.2f} GB")
    
    return eeg_out, video_out, latents_out, total_segments


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient concatenation of preprocessed .npz files"
    )
    
    parser.add_argument('--input-dir', required=True,
                       help='Directory containing .npz files')
    parser.add_argument('--task', required=True,
                       help='Task name (e.g., dme, tp, inscapes)')
    parser.add_argument('--pattern', default='*.npz',
                       help='File pattern to match (default: *.npz)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    if not input_dir.exists():
        print(f"❌ Error: Directory {input_dir} does not exist!")
        return
    
    # Find all .npz files
    npz_files = sorted(input_dir.glob(args.pattern))
    
    if len(npz_files) == 0:
        print(f"❌ Error: No .npz files found in {input_dir}")
        return
    
    print(f"Found {len(npz_files)} .npz files:")
    for f in npz_files:
        print(f"  - {f.name}")
    
    # Concatenate
    eeg_out, video_out, latents_out, total_segments = concatenate_with_memmap(
        npz_files,
        input_dir,
        args.task,
    )
    
    # Save metadata
    metadata = {
        'task': args.task,
        'num_files': len(npz_files),
        'num_segments': int(total_segments),
        'files': [f.name for f in npz_files],
    }
    
    meta_out = input_dir / f"{args.task}_metadata.json"
    with open(meta_out, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n  ✓ Saved metadata to: {meta_out}")
    
    print(f"\n{'='*70}")
    print("✅ CONCATENATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Total segments: {total_segments}")
    print(f"Files processed: {len(npz_files)}")


if __name__ == "__main__":
    main()
