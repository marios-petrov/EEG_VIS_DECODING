#!/usr/bin/env python3
"""
Generate BLIP captions for preprocessed video frames
Runs on existing dme_video_frames.npy
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPTextModel,
    CLIPTokenizer,
)


def generate_captions_and_embeddings(
    video_segments: np.ndarray,
    movie_title: str,
    device: torch.device,
    batch_size: int = 8,
):
    """
    Generate BLIP captions and CLIP embeddings for video segments
    Memory-efficient version that processes in batches
    
    Args:
        video_segments: [num_segments, num_frames, H, W, 3] uint8 array
        movie_title: Movie title (e.g., "Despicable Me", "The Present")
        device: Device to use
        batch_size: Number of segments to process at once
    """
    print(f"📝 Generating captions...")
    print(f"   Segments: {video_segments.shape[0]:,}")
    print(f"   Frames per segment: {video_segments.shape[1]}")
    print(f"   Movie: '{movie_title}'")
    
    # Load BLIP
    print(f"\n🎨 Loading BLIP model...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to(device)
    blip_model.eval()
    
    # Load CLIP
    print(f"🎨 Loading CLIP text encoder...")
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_text_encoder.eval()
    
    num_segments = video_segments.shape[0]
    num_frames = video_segments.shape[1]
    
    all_captions = []
    all_embeddings = []
    
    # Process in batches to avoid OOM
    for seg_idx in tqdm(range(0, num_segments, batch_size), desc="Captioning"):
        end_idx = min(seg_idx + batch_size, num_segments)
        batch_segments = video_segments[seg_idx:end_idx]
        
        batch_captions = []
        batch_embeddings = []
        
        for segment in batch_segments:
            # Caption the middle frame
            mid_frame_idx = num_frames // 2
            frame = segment[mid_frame_idx]
            
            # Convert to PIL
            pil_image = Image.fromarray(frame)
            
            # Generate base caption with BLIP (no initial context)
            inputs = blip_processor(pil_image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                out = blip_model.generate(**inputs, max_length=50, num_beams=3)
                base_caption = blip_processor.decode(out[0], skip_special_tokens=True)
            
            # Format as: "A scene from the animated movie [Movie Title] in which [base_caption]"
            full_caption = f"A scene from the animated movie {movie_title} in which {base_caption}"
            
            batch_captions.append(full_caption)
            
            # Generate CLIP embeddings for all frames using the full caption
            seg_embeddings = []
            for frame in segment:
                # Use full caption as text prompt
                text_inputs = clip_tokenizer(
                    full_caption,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)
                
                with torch.no_grad():
                    text_embedding = clip_text_encoder(**text_inputs).last_hidden_state
                    # Use [CLS] token
                    text_embedding = text_embedding[:, 0:1, :]  # [1, 1, 1024]
                
                seg_embeddings.append(text_embedding.cpu().numpy())
            
            batch_embeddings.append(np.stack(seg_embeddings, axis=0))  # [num_frames, 1, 1, 1024]
        
        all_captions.extend(batch_captions)
        all_embeddings.extend(batch_embeddings)
        
        # Free memory
        torch.cuda.empty_cache()
    
    all_embeddings = np.stack(all_embeddings, axis=0)  # [segments, frames, 1, 1, 1024]
    
    print(f"\n✓ Generated {len(all_captions):,} captions")
    print(f"✓ Embeddings shape: {all_embeddings.shape}")
    
    return all_captions, all_embeddings


def main():
    parser = argparse.ArgumentParser(description="Generate captions for preprocessed video frames")
    
    parser.add_argument('--preprocessed-dir', required=True,
                       help='Directory containing preprocessed data')
    parser.add_argument('--task', required=True, choices=['dme', 'tp', 'inscapes'],
                       help='Task name')
    parser.add_argument('--movie-title', default=None,
                       help='Movie title (e.g., "Despicable Me", "The Present"). Auto-detected from task if not specified.')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for processing')
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    
    # Auto-detect movie title if not specified
    if args.movie_title is None:
        movie_titles = {
            'dme': 'Despicable Me',
            'tp': 'The Present',
            'inscapes': 'Inscapes'
        }
        args.movie_title = movie_titles.get(args.task, 'an animated movie')
    
    preprocessed_dir = Path(args.preprocessed_dir)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("GENERATING CAPTIONS FOR PREPROCESSED DATA")
    print("=" * 70)
    
    # Load video frames (memory-mapped to avoid loading into RAM)
    video_path = preprocessed_dir / f"{args.task}_video_frames.npy"
    
    if not video_path.exists():
        print(f"❌ Error: {video_path} not found!")
        return
    
    print(f"\n📂 Loading video frames (memory-mapped)...")
    video_frames = np.load(video_path, mmap_mode='r')
    print(f"   Shape: {video_frames.shape}")
    print(f"   Size: {video_path.stat().st_size / 1e9:.2f} GB")
    
    # Generate captions
    captions, embeddings = generate_captions_and_embeddings(
        video_frames,
        args.movie_title,
        device,
        batch_size=args.batch_size,
    )
    
    # Save results
    captions_file = preprocessed_dir / f"{args.task}_captions_hd.json"
    embeddings_file = preprocessed_dir / f"{args.task}_clip_text_embeddings.npy"
    
    print(f"\n💾 Saving results...")
    with open(captions_file, 'w') as f:
        json.dump({'captions': captions}, f, indent=2)
    
    np.save(embeddings_file, embeddings)
    
    print(f"   ✓ Captions: {captions_file}")
    print(f"   ✓ Embeddings: {embeddings_file}")
    
    # Show some examples
    print(f"\n📝 Sample captions:")
    for i in range(min(5, len(captions))):
        print(f"   {i}: {captions[i]}")
    
    print("\n" + "=" * 70)
    print("✅ CAPTION GENERATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
