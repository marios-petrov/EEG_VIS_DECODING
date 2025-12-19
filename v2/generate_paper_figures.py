#!/usr/bin/env python3
"""
Generate paper figure samples using existing blurry images.

Usage:
    python generate_paper_figures.py --method blip --output-dir ./paper_images
    python generate_paper_figures.py --method gt_caption --output-dir ./paper_images
"""

import os
os.environ['HF_HOME'] = '/local-scratch/marios-datasets/hf-cache'

import argparse
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Selected samples for paper figure (10 samples: 5 good, 5 bad)
# Same samples for both BLIP and GT caption methods
PAPER_SAMPLES = [
    # Good (categories that work well - high accuracy)
    {"idx": 0, "category": "tree"},
    {"idx": 1, "category": "flower"},
    {"idx": 2, "category": "jellyfish"},
    {"idx": 3, "category": "cat"},
    {"idx": 4, "category": "road"},
    # Bad (categories that struggle - low accuracy)
    {"idx": 5, "category": "guitar"},
    {"idx": 6, "category": "drum"},
    {"idx": 7, "category": "shark"},
    {"idx": 8, "category": "airplane"},
    {"idx": 9, "category": "piano"},
]


def load_blip(device):
    """Load BLIP for image captioning"""
    from transformers import BlipProcessor, BlipForConditionalGeneration
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        torch_dtype=torch.float16,
    ).to(device).eval()
    
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


def clean_caption(caption: str) -> str:
    """
    Clean BLIP caption by removing:
    - "blurry", "blurred", "blur"
    - "there is a picture of", "this is a picture of"
    - "there is a photo of", "this is a photo of"
    - "there is a/an", "this is a/an" at the start
    - Double spaces and cleanup
    """
    import re
    
    # Convert to lowercase for processing
    cleaned = caption.lower().strip()
    
    # Remove "blurry/blurred/blur" and related
    cleaned = re.sub(r'\b(blurry|blurred|blur|fuzzy|hazy)\b', '', cleaned)
    
    # Remove "there is a picture/photo/image of"
    cleaned = re.sub(r'^there is (a |an )?(picture|photo|image) of (a |an )?', '', cleaned)
    cleaned = re.sub(r'^this is (a |an )?(picture|photo|image) of (a |an )?', '', cleaned)
    
    # Remove standalone "picture of", "photo of", "image of"
    cleaned = re.sub(r'\b(picture|photo|image) of (a |an )?', '', cleaned)
    
    # Remove "there is a/an" or "this is a/an" at start
    cleaned = re.sub(r'^there is (a |an )?', '', cleaned)
    cleaned = re.sub(r'^this is (a |an )?', '', cleaned)
    
    # Clean up multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Capitalize first letter
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    
    # If caption became too short or empty, return a generic description
    if len(cleaned) < 3:
        return "A scene"
    
    return cleaned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=['blip', 'gt_caption'])
    parser.add_argument('--blurry-dir', type=str, 
                       default='/local-scratch/marios-datasets/SEED/eval_v2/stage1_merged/blurry')
    parser.add_argument('--metadata', type=str,
                       default='/local-scratch/marios-datasets/SEED/eval_v2/stage1_merged/metadata.json')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--strength', type=float, default=0.5)
    parser.add_argument('--num-inference-steps', type=int, default=20)
    parser.add_argument('--guidance-scale', type=float, default=4.5)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    blurry_dir = Path(args.blurry_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"PAPER FIGURE GENERATION - {args.method.upper()}")
    print("=" * 60)
    
    # Load metadata to get GT captions/concepts
    with open(args.metadata) as f:
        metadata = json.load(f)
    samples = metadata.get('samples', [])
    
    # Build lookup by sample_idx
    sample_lookup = {s['sample_idx']: s for s in samples}
    
    print(f"Blurry dir: {blurry_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Method: {args.method}")
    print(f"Total metadata samples: {len(samples)}")
    
    # Load SD3 pipeline
    print("\nLoading SD3 pipeline...")
    from diffusers import StableDiffusion3Img2ImgPipeline
    sd3_pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.float16,
    ).to(device)
    sd3_pipe.set_progress_bar_config(disable=True)
    print("  ✓ SD3 loaded")
    
    # Load BLIP if needed
    blip_processor, blip_model = None, None
    if args.method == 'blip':
        print("Loading BLIP...")
        blip_processor, blip_model = load_blip(device)
        print("  ✓ BLIP loaded")
    
    print(f"\nProcessing {len(PAPER_SAMPLES)} samples...")
    print("=" * 60)
    
    results = []
    
    for sample in tqdm(PAPER_SAMPLES):
        idx = sample['idx']
        category = sample['category']
        
        # Find the actual sample index in metadata that matches this category
        # We need to find samples of this category
        matching_samples = [s for s in samples if s.get('concept_name', '').lower() == category.lower()]
        
        if not matching_samples:
            # Try concept_id based lookup
            matching_samples = [s for s in samples]  # Just use first N samples as fallback
        
        if idx >= len(matching_samples):
            print(f"  Warning: Not enough samples for {category}, using idx {idx % len(samples)}")
            actual_sample = samples[idx % len(samples)]
        else:
            # Find a sample of this category
            category_samples = [s for s in samples if category.lower() in s.get('concept_name', '').lower()]
            if category_samples:
                actual_sample = category_samples[0]
            else:
                actual_sample = samples[idx]
        
        sample_idx = actual_sample['sample_idx']
        gt_concept = actual_sample.get('concept_name', actual_sample.get('gt_concept_name', category))
        
        # Load blurry image
        blurry_path = blurry_dir / f"{sample_idx:05d}.png"
        if not blurry_path.exists():
            print(f"  Warning: {blurry_path} not found, skipping")
            continue
        
        blurry_img = Image.open(blurry_path).convert("RGB")
        
        # Get caption based on method
        if args.method == 'blip':
            raw_caption = caption_image(blurry_img, blip_processor, blip_model, device)
            caption = clean_caption(raw_caption)
            print(f"  [{idx}] {category}: BLIP \"{raw_caption}\" -> \"{caption}\"")
        else:  # gt_caption
            caption = f"a video of a {gt_concept}"
            print(f"  [{idx}] {category}: GT -> \"{caption}\"")
        
        # Refine with SD3
        with torch.no_grad():
            refined = sd3_pipe(
                prompt=caption,
                image=blurry_img,
                strength=args.strength,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            ).images[0]
        
        # Save
        output_filename = f"{idx:02d}_{category}_{args.method}.png"
        refined.save(output_dir / output_filename)
        
        # Also copy blurry and save GT if this is first method run
        blurry_out = output_dir / f"{idx:02d}_{category}_blurry.png"
        if not blurry_out.exists():
            blurry_img.save(blurry_out)
        
        results.append({
            'idx': idx,
            'category': category,
            'sample_idx': sample_idx,
            'caption': caption,
            'method': args.method,
        })
    
    # Save results
    results_path = output_dir / f"results_{args.method}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"DONE - {args.method.upper()}")
    print(f"{'=' * 60}")
    print(f"Saved {len(results)} images to {output_dir}")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
