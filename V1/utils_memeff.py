#!/usr/bin/env python3
"""
Memory-efficient dataset for UNet fine-tuning
Uses memory-mapped numpy arrays to avoid loading 38GB into RAM
Perfect for systems with limited RAM but lots of disk space
"""

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class LatentFramesWithPrompts_MemEff(Dataset):
    """
    Memory-efficient dataset for fine-tuning UNet with latents and text prompts
    Uses numpy memory-mapping to stream data from disk without loading into RAM
    """
    
    def __init__(
        self,
        preprocessed_dir: Path,
        task: str,
        style_suffix: str = "animated movie still, cinematic lighting, high detail",
        flow_weighting: bool = True,
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.task = task
        
        # Memory-map latents (CRITICAL: Don't load into RAM!)
        latents_path = preprocessed_dir / f"{task}_vae_latents_hd.npy"
        print(f"📂 Memory-mapping latents from: {latents_path}")
        self.latents_mmap = np.load(latents_path, mmap_mode='r')
        
        # Get shape info
        S, F = self.latents_mmap.shape[0], self.latents_mmap.shape[1]
        self.num_segments = S
        self.frames_per_segment = F
        self.total_frames = S * F
        
        print(f"✓ Dataset loaded (memory-mapped, not in RAM):")
        print(f"   Segments: {S:,}")
        print(f"   Frames/segment: {F}")
        print(f"   Total frames: {self.total_frames:,}")
        print(f"   Latent shape per frame: {self.latents_mmap.shape[2:]}")
        
        # Load captions (small, ~few KB, OK to keep in RAM)
        captions_path = preprocessed_dir / f"{task}_captions_hd.json"
        if captions_path.exists():
            with open(captions_path, "r") as f:
                data = json.load(f)
                caps = data.get("captions", [])
        else:
            caps = []
        
        if not caps or len(caps) != S:
            caps = [""] * S
        
        # Load optical flow scores (small array, OK to keep in RAM)
        flows = None
        flow_path = preprocessed_dir / f"{task}_flow_scores_hd.npy"
        if flow_path.exists() and flow_weighting:
            flows = np.load(flow_path)
            if flows.size > 0:
                fmin, fmax = float(flows.min()), float(flows.max())
                flows = (flows - fmin) / (fmax - fmin) if fmax > fmin else np.zeros_like(flows)
        
        # Build prompts (keep in RAM, just strings, minimal memory)
        seg_prompts = []
        for si in range(S):
            cap = (caps[si] or "").strip()
            base = f"{cap}. {style_suffix}" if cap else style_suffix
            
            # Add motion prompt for high-flow segments
            if flows is not None and flows.size == S and flows[si] > 0.66:
                base = base + ", dynamic motion, action shot"
            
            seg_prompts.append(base)
        
        # Expand prompts for all frames (just strings, minimal memory)
        self.prompts = []
        for si in range(S):
            self.prompts.extend([seg_prompts[si]] * F)
        
        assert len(self.prompts) == self.total_frames
        
        print(f"✓ Built {len(self.prompts):,} prompts")
    
    def __len__(self):
        return self.total_frames
    
    def __getitem__(self, idx):
        """
        Fetch a single frame on-demand from disk
        This is where memory-mapping shines - only loads what's needed!
        """
        # Calculate which segment and frame
        seg_idx = idx // self.frames_per_segment
        frame_idx = idx % self.frames_per_segment
        
        # Load ONLY this frame from disk (memory-mapped access)
        # This doesn't load the entire 38GB file!
        latent = self.latents_mmap[seg_idx, frame_idx].copy()  # copy to get writable array
        latent = torch.from_numpy(latent).float()  # [4, 96, 96]
        
        prompt = self.prompts[idx]
        
        return latent, prompt


if __name__ == "__main__":
    # Test the dataset
    import sys
    if len(sys.argv) > 1:
        preprocessed_dir = Path(sys.argv[1])
        task = sys.argv[2] if len(sys.argv) > 2 else "dme"
        
        print("=" * 70)
        print("TESTING MEMORY-EFFICIENT DATASET")
        print("=" * 70)
        
        dataset = LatentFramesWithPrompts_MemEff(
            preprocessed_dir=preprocessed_dir,
            task=task,
        )
        
        print(f"\n📊 Dataset size: {len(dataset):,} frames")
        
        # Test fetching a few samples
        print("\n🔍 Testing sample access...")
        for i in [0, 100, 1000]:
            latent, prompt = dataset[i]
            print(f"  Sample {i}: latent shape={latent.shape}, prompt length={len(prompt)}")
        
        print("\n✅ Memory-efficient dataset working correctly!")
