#!/usr/bin/env python3
"""
BLIP-on-Blurry Refinement: Fair EEG-to-Video pipeline.

Pipeline:
    Blurry image (from EEG) → BLIP caption → SD3 refinement

This is FAIR because:
- No GT information used
- Caption describes what BLIP sees in YOUR prediction
- Tests if SD3 can enhance when given predicted content

Usage:
    python eval_blip_refine.py \
        --blurry-dir /path/to/stage1_merged/blurry \
        --metadata /path/to/stage1_merged/metadata.json \
        --output-dir /path/to/blip_refined
"""

import os
os.environ['HF_HOME'] = '/local-scratch/marios-datasets/hf-cache'

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image


def load_blip():
    """Load BLIP for image captioning"""
    from transformers import BlipProcessor, BlipForConditionalGeneration
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        torch_dtype=torch.float16,
    )
    return processor, model


def caption_image(image: Image.Image, processor, model, device) -> str:
    """Generate caption for image using BLIP"""
    inputs = processor(image, return_tensors="pt").to(device, torch.float16)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=50,
            num_beams=5,
            early_stopping=True,
        )
    
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blurry-dir', type=str, required=True,
                       help='Directory with blurry images')
    parser.add_argument('--metadata', type=str, required=True,
                       help='Metadata JSON from stage1')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--start-idx', type=int, default=0,
                       help='Start sample index (for parallel processing)')
    parser.add_argument('--end-idx', type=int, default=None,
                       help='End sample index (for parallel processing)')
    parser.add_argument('--device', type=str, default='cuda')
    # SD3 settings
    parser.add_argument('--strength', type=float, default=0.5)
    parser.add_argument('--num-inference-steps', type=int, default=20)
    parser.add_argument('--guidance-scale', type=float, default=4.5)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("BLIP-ON-BLURRY REFINEMENT (Fair Pipeline)")
    print("="*60)
    
    blurry_dir = Path(args.blurry_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "refined").mkdir(exist_ok=True)
    
    # Load metadata
    with open(args.metadata) as f:
        metadata = json.load(f)
    samples = metadata.get('samples', [])
    
    print(f"Blurry dir: {blurry_dir}")
    print(f"Samples: {len(samples)}")
    
    # Load models
    print("\nLoading models...")
    
    # BLIP
    blip_processor, blip_model = load_blip()
    blip_model = blip_model.to(device)
    blip_model.eval()
    print("  ✓ BLIP loaded")
    
    # SD3 img2img
    from diffusers import StableDiffusion3Img2ImgPipeline
    sd3_pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.float16,
    ).to(device)
    sd3_pipe.set_progress_bar_config(disable=True)
    print("  ✓ SD3 pipeline loaded")
    
    print(f"\nSettings:")
    print(f"  Strength: {args.strength}")
    print(f"  Steps: {args.num_inference_steps}")
    print(f"  Guidance: {args.guidance_scale}")
    print(f"  Caption source: BLIP on blurry image")
    print("="*60 + "\n")
    
    # Process
    new_metadata = []
    
    # Apply index range
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx else len(samples)
    if args.max_samples:
        end_idx = min(start_idx + args.max_samples, end_idx)
    
    samples_to_process = samples[start_idx:end_idx]
    print(f"Processing samples {start_idx} to {end_idx} ({len(samples_to_process)} samples)")
    
    for i, sample in enumerate(tqdm(samples_to_process)):
        idx = sample['sample_idx']
        
        # Load blurry image
        blurry_path = blurry_dir / f"{idx:05d}.png"
        if not blurry_path.exists():
            continue
        
        blurry_img = Image.open(blurry_path).convert("RGB")
        
        # Generate caption with BLIP
        caption = caption_image(blurry_img, blip_processor, blip_model, device)
        
        # Refine with SD3
        with torch.no_grad():
            refined_result = sd3_pipe(
                prompt=caption,
                image=blurry_img,
                strength=args.strength,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            ).images[0]
        
        # Save
        refined_result.save(output_dir / "refined" / f"{idx:05d}.png")
        
        # Update metadata
        new_sample = sample.copy()
        new_sample['blip_caption'] = caption
        new_metadata.append(new_sample)
        
        # Progress
        if (i + 1) % 100 == 0:
            print(f"  Sample {i+1}: \"{caption[:50]}...\"")
    
    # Save metadata (partial for this run)
    output_meta_path = output_dir / f"metadata_{start_idx}_{end_idx}.json"
    with open(output_meta_path, 'w') as f:
        json.dump({
            'settings': {
                'strength': args.strength,
                'num_inference_steps': args.num_inference_steps,
                'guidance_scale': args.guidance_scale,
                'caption_source': 'blip_on_blurry',
                'start_idx': start_idx,
                'end_idx': end_idx,
            },
            'samples': new_metadata,
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("BLIP REFINEMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Processed: {len(new_metadata)} samples ({start_idx}-{end_idx})")
    print(f"Output: {output_dir}")
    print(f"Metadata: {output_meta_path}")
    print(f"\nSample captions:")
    for s in new_metadata[:5]:
        print(f"  [{s['concept_id']:2d}] {s.get('blip_caption', 'N/A')[:60]}")


if __name__ == "__main__":
    main()
