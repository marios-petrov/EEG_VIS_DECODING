#!/usr/bin/env python3
"""
Stage 1: Refine blurry EEG-predicted frames using SD3 img2img.

Saves refined frames to disk, then run stage 2 (SVD) separately.

Usage:
    # Stage 1: Refine frames
    python eval_stage1_refine.py \
        --checkpoint /path/to/adapter_best.pt \
        --preprocessed-dir /path/to/preprocessed \
        --output-dir /path/to/eval_results \
        --max-samples 100
    
    # Stage 2: Generate videos (separate script)
    python eval_stage2_svd.py \
        --refined-dir /path/to/eval_results/refined_frames \
        --output-dir /path/to/eval_results
"""

import os
os.environ['HF_HOME'] = '/local-scratch/marios-datasets/hf-cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/local-scratch/marios-datasets/hf-cache/hub'

import argparse
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import cv2
from typing import List, Dict

import sys
sys.path.append('/home/mpetrov/EEG_Reconstruction')


SEED_DV_CONCEPTS = [
    "hot air balloon", "roller coaster", "drone racing", "wing suit flying", "ski lift",
    "dogs playing", "cats playing", "underwater diving", "safari animals", "horse riding",
    "fireworks display", "concert performance", "street dance", "magic show", "circus act",
    "cooking show", "sports highlights", "news broadcast", "weather forecast", "talk show",
    "car racing", "motorcycle riding", "boat sailing", "train journey", "airplane takeoff",
    "mountain climbing", "surfing waves", "skateboarding", "parkour running", "bungee jumping",
    "painting art", "sculpture making", "pottery crafting", "glass blowing", "woodworking",
    "science experiment", "robot demonstration", "space footage", "nature documentary", "city timelapse"
]


class SEEDDVDataset(Dataset):
    """SEED-DV test dataset"""
    
    def __init__(
        self,
        preprocessed_dir: Path,
        subject_ids: List[int],
        precomputed_embeddings: Dict[str, torch.Tensor] = None,
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.precomputed_embeddings = precomputed_embeddings
        self.samples = []
        
        self.eeg_data = {}
        self.latent_data = {}
        
        for subj_id in subject_ids:
            subj_dir = self.preprocessed_dir / f"sub{subj_id}"
            if not subj_dir.exists():
                subj_dir = self.preprocessed_dir / f"subject_{subj_id:02d}"
            if not subj_dir.exists():
                print(f"Warning: Subject {subj_id} not found")
                continue
            
            eeg_path = subj_dir / "eeg.npy"
            if not eeg_path.exists():
                eeg_path = subj_dir / "eeg_segments.npy"
            
            latent_path = subj_dir / "latents.npy"
            if not latent_path.exists():
                latent_path = subj_dir / "video_latents.npy"
            
            metadata_path = subj_dir / "metadata.json"
            
            if not all(p.exists() for p in [eeg_path, latent_path, metadata_path]):
                continue
            
            self.eeg_data[subj_id] = np.load(eeg_path)
            self.latent_data[subj_id] = np.load(latent_path)
            
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            for seg in metadata['segments']:
                if not seg['is_train']:
                    seg['subject_id'] = subj_id
                    self.samples.append(seg)
        
        print(f"  Loaded {len(self.samples)} test samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        seg = self.samples[idx]
        subj_id = seg['subject_id']
        seg_idx = seg['segment_idx']
        
        eeg = self.eeg_data[subj_id][seg_idx]
        all_latents = self.latent_data[subj_id][seg_idx]
        
        if self.precomputed_embeddings is not None:
            caption = seg['caption']
            caption_emb = self.precomputed_embeddings.get(caption, torch.zeros(2048))
        else:
            caption_emb = torch.zeros(2048)
        
        return {
            'eeg': torch.from_numpy(eeg).float(),
            'all_latents': torch.from_numpy(all_latents).float(),
            'caption_emb': caption_emb,
            'subject_id': subj_id,
            'concept_id': seg['concept_id'],
            'caption': seg['caption'],
            'segment_idx': seg_idx,
        }


def load_vae():
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    vae.eval()
    return vae


def decode_latent(latent: torch.Tensor, vae, device) -> np.ndarray:
    if latent.dim() == 3:
        latent = latent.unsqueeze(0)
    
    latent = latent.to(device, dtype=torch.float16)
    latent = (latent / vae.config.scaling_factor) + vae.config.shift_factor
    
    with torch.no_grad():
        image = vae.decode(latent, return_dict=False)[0]
    
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    
    return image[0]


def decode_latents_to_frames(latents: torch.Tensor, vae, device) -> List[np.ndarray]:
    frames = []
    for i in range(len(latents)):
        frame = decode_latent(latents[i], vae, device)
        frames.append(frame)
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--preprocessed-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--subjects', type=str, default='1-20')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no-captions', action='store_true')
    # SD3 refinement settings
    parser.add_argument('--strength', type=float, default=0.5)
    parser.add_argument('--num-inference-steps', type=int, default=20)
    parser.add_argument('--guidance-scale', type=float, default=4.5)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Parse subjects
    if '-' in args.subjects:
        start, end = map(int, args.subjects.split('-'))
        subject_ids = list(range(start, end + 1))
    else:
        subject_ids = [int(s) for s in args.subjects.split(',')]
    
    print(f"Stage 1: Refining frames for subjects {subject_ids}")
    
    # Setup output directory
    use_captions = not args.no_captions
    caption_str = "with_captions" if use_captions else "no_captions"
    output_dir = Path(args.output_dir) / f"stage1_{caption_str}_str{args.strength}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "blurry").mkdir(exist_ok=True)
    (output_dir / "refined").mkdir(exist_ok=True)
    (output_dir / "gt_frames").mkdir(exist_ok=True)
    
    # Load models
    print("\nLoading models...")
    vae = load_vae().to(device)
    
    # Load SD3 img2img
    from diffusers import StableDiffusion3Img2ImgPipeline
    print("Loading SD3 img2img pipeline...")
    sd3_pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.float16,
    ).to(device)
    sd3_pipe.set_progress_bar_config(disable=True)
    print("  ✓ SD3 pipeline loaded")
    
    # Load EEGMamba adapter
    from eegmamba_adapter_optimized import EEGMambaAdapter
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = EEGMambaAdapter(
        eeg_channels=62,
        eeg_samples=400,
        d_model=512,
        n_layers=4,
        latent_channels=16,
        latent_height=96,
        latent_width=96,
        text_embed_dim=2048,
    ).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"  ✓ EEGMamba loaded (epoch {checkpoint.get('epoch', '?')})")
    
    # Load caption embeddings
    checkpoint_dir = Path(args.checkpoint).parent
    embeddings_path = checkpoint_dir / "caption_embeddings.pt"
    if not embeddings_path.exists():
        embeddings_path = Path(args.preprocessed_dir).parent / "caption_embeddings.pt"
    
    caption_embeddings = None
    if embeddings_path.exists():
        caption_embeddings = torch.load(embeddings_path, weights_only=False)
        print(f"  ✓ Loaded {len(caption_embeddings)} caption embeddings")
    
    # Create dataset
    test_dataset = SEEDDVDataset(
        Path(args.preprocessed_dir),
        subject_ids,
        precomputed_embeddings=caption_embeddings,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    
    print(f"\n{'='*60}")
    print(f"Stage 1: SD3 Refinement")
    print(f"  Strength: {args.strength}")
    print(f"  Steps: {args.num_inference_steps}")
    print(f"  Guidance: {args.guidance_scale}")
    print(f"  Captions: {'enabled' if use_captions else 'DISABLED'}")
    print(f"{'='*60}\n")
    
    # Process samples
    metadata_list = []
    max_samples = args.max_samples or len(test_dataset)
    
    for batch_idx, batch in enumerate(tqdm(test_loader, total=min(max_samples, len(test_loader)))):
        if batch_idx >= max_samples:
            break
        
        eeg = batch['eeg'].to(device)
        all_latents = batch['all_latents'][0]
        caption_emb = batch['caption_emb'].to(device)
        concept_id = batch['concept_id'].item()
        subject_id = batch['subject_id'].item()
        caption = batch['caption'][0]
        
        if not use_captions:
            caption_emb = torch.zeros_like(caption_emb)
        
        # Generate blurry prediction from EEG
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred_latent = model(eeg, caption_emb)
        
        # Decode to blurry image
        blurry_img = decode_latent(pred_latent[0], vae, device)
        
        # Refine with SD3
        prompt = caption if use_captions else ""
        pil_blurry = Image.fromarray(blurry_img).convert("RGB")
        
        with torch.no_grad():
            refined_result = sd3_pipe(
                prompt=prompt,
                image=pil_blurry,
                strength=args.strength,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            ).images[0]
        
        refined_img = np.array(refined_result)
        
        # Decode GT frames
        gt_frames = decode_latents_to_frames(all_latents, vae, device)
        
        # Save images
        Image.fromarray(blurry_img).save(output_dir / "blurry" / f"{batch_idx:05d}.png")
        Image.fromarray(refined_img).save(output_dir / "refined" / f"{batch_idx:05d}.png")
        
        # Save GT frames as subfolder
        gt_dir = output_dir / "gt_frames" / f"{batch_idx:05d}"
        gt_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(gt_frames):
            Image.fromarray(frame).save(gt_dir / f"frame_{i:03d}.png")
        
        # Store metadata
        metadata_list.append({
            'sample_idx': batch_idx,
            'subject_id': subject_id,
            'concept_id': concept_id,
            'concept_name': SEED_DV_CONCEPTS[concept_id],
            'caption': caption,
            'n_gt_frames': len(gt_frames),
        })
    
    # Save metadata
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump({
            'settings': {
                'strength': args.strength,
                'num_inference_steps': args.num_inference_steps,
                'guidance_scale': args.guidance_scale,
                'use_captions': use_captions,
            },
            'samples': metadata_list,
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Stage 1 Complete!")
    print(f"  Processed: {len(metadata_list)} samples")
    print(f"  Output: {output_dir}")
    print(f"\nNext: Run stage 2 (SVD) with:")
    print(f"  python eval_stage2_svd.py --stage1-dir {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
