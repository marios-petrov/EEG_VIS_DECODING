#!/usr/bin/env python3
"""
Evaluate EEGMamba + SD 3.5 on SEED-DV dataset (Diffusion Refinement).

Uses SD 3.5 img2img pipeline to refine blurry predicted images into sharp ones.

Usage:
    python eval_eegmamba_seed_dv_diffusion.py \
        --checkpoint /path/to/adapter_best.pt \
        --preprocessed-dir /path/to/preprocessed \
        --output-dir /path/to/eval_results \
        --strength 0.5 \
        --num-inference-steps 20
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
from collections import defaultdict
from typing import List, Dict, Tuple
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import sys
sys.path.append('/home/mpetrov/EEG_Reconstruction')


# SEED-DV concepts (40 classes)
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


class SEEDDVTestDataset(Dataset):
    """SEED-DV test dataset (block 7 only)"""
    
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
                subj_dir = self.preprocessed_dir / f"subject_{subj_id}"
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
        latent = self.latent_data[subj_id][seg_idx]
        target_latent = latent[len(latent)//2]
        
        if self.precomputed_embeddings is not None:
            caption = seg['caption']
            caption_emb = self.precomputed_embeddings.get(caption, torch.zeros(2048))
        else:
            caption_emb = torch.zeros(2048)
        
        return {
            'eeg': torch.from_numpy(eeg).float(),
            'latent': torch.from_numpy(target_latent).float(),
            'caption_emb': caption_emb,
            'subject_id': subj_id,
            'concept_id': seg['concept_id'],
            'caption': seg['caption'],
        }


def load_vae():
    """Load SD 3.5 VAE"""
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    vae.eval()
    return vae


def decode_latent(latent: torch.Tensor, vae, device) -> np.ndarray:
    """Decode latent to image"""
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


def load_sd3_img2img_pipeline(device):
    """Load SD 3.5 img2img pipeline"""
    from diffusers import StableDiffusion3Img2ImgPipeline
    
    print("Loading SD 3.5 img2img pipeline...")
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    print("  ✓ Pipeline loaded")
    
    return pipe


def refine_image(
    pipe,
    blurry_image: np.ndarray,
    prompt: str,
    strength: float = 0.5,
    num_inference_steps: int = 20,
    guidance_scale: float = 4.5,
) -> np.ndarray:
    """Refine a blurry image using SD 3.5 img2img"""
    
    # Convert to PIL
    pil_image = Image.fromarray(blurry_image).convert("RGB")
    
    # Run img2img
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=pil_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
    
    return np.array(result)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
    return ssim(img1_gray, img2_gray, data_range=255)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    return psnr(img1, img2, data_range=255)


def load_clip_classifier(device):
    from transformers import CLIPProcessor, CLIPModel
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        torch_dtype=torch.float16,
    ).to(device).eval()
    return model, processor


def classify_image_clip(
    image: np.ndarray,
    model,
    processor,
    device,
    concepts: List[str] = SEED_DV_CONCEPTS,
) -> Tuple[int, np.ndarray]:
    pil_image = Image.fromarray(image)
    text_prompts = [f"a video frame of {concept}" for concept in concepts]
    
    inputs = processor(
        text=text_prompts,
        images=pil_image,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]
        probs = torch.softmax(logits, dim=0).cpu().numpy()
    
    return int(np.argmax(probs)), probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--preprocessed-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--subjects', type=str, default='1-20')
    parser.add_argument('--batch-size', type=int, default=1, help='Must be 1 for img2img')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no-captions', action='store_true')
    parser.add_argument('--strength', type=float, default=0.5,
                       help='Img2img strength (0=no change, 1=full regeneration)')
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
    
    print(f"Evaluating on subjects: {subject_ids}")
    
    # Load models
    print("\nLoading models...")
    vae = load_vae().to(device)
    clip_model, clip_processor = load_clip_classifier(device)
    pipe = load_sd3_img2img_pipeline(device)
    
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
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load caption embeddings
    checkpoint_dir = Path(args.checkpoint).parent
    embeddings_path = checkpoint_dir / "caption_embeddings.pt"
    if not embeddings_path.exists():
        embeddings_path = Path(args.preprocessed_dir).parent / "caption_embeddings.pt"
    
    caption_embeddings = None
    if embeddings_path.exists():
        caption_embeddings = torch.load(embeddings_path, weights_only=False)
        print(f"Loaded {len(caption_embeddings)} caption embeddings")
    
    # Create dataset
    test_dataset = SEEDDVTestDataset(
        Path(args.preprocessed_dir),
        subject_ids,
        precomputed_embeddings=caption_embeddings,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Must be 1 for img2img
        shuffle=False,
        num_workers=0,
    )
    
    # Setup output directory
    use_captions = not args.no_captions
    caption_str = "with_captions" if use_captions else "no_captions"
    output_dir = Path(args.output_dir) / f"{caption_str}_diffusion_str{args.strength}_steps{args.num_inference_steps}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    (output_dir / "ground_truth").mkdir(exist_ok=True)
    (output_dir / "comparisons").mkdir(exist_ok=True)
    (output_dir / "blurry").mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Diffusion refinement settings:")
    print(f"  Strength: {args.strength}")
    print(f"  Steps: {args.num_inference_steps}")
    print(f"  Guidance: {args.guidance_scale}")
    print(f"  Captions: {'enabled' if use_captions else 'DISABLED'}")
    print(f"{'='*60}\n")
    
    # Run evaluation
    all_results = []
    correct_40way = 0
    correct_2way = 0
    correct_top5 = 0
    total_samples = 0
    
    max_samples = args.max_samples or len(test_dataset)
    
    for batch_idx, batch in enumerate(tqdm(test_loader, total=min(max_samples, len(test_loader)))):
        if batch_idx >= max_samples:
            break
        
        eeg = batch['eeg'].to(device)
        target_latent = batch['latent'].to(device)
        caption_emb = batch['caption_emb'].to(device)
        concept_id = batch['concept_id'].item()
        subject_id = batch['subject_id'].item()
        caption = batch['caption'][0]
        
        if not use_captions:
            caption_emb = torch.zeros_like(caption_emb)
        
        # Generate blurry prediction
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred_latent = model(eeg, caption_emb)
        
        # Decode to blurry image
        blurry_img = decode_latent(pred_latent[0], vae, device)
        gt_img = decode_latent(target_latent[0], vae, device)
        
        # Refine with diffusion
        prompt = caption if use_captions else ""
        refined_img = refine_image(
            pipe,
            blurry_img,
            prompt=prompt,
            strength=args.strength,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )
        
        # Compute metrics
        ssim_val = compute_ssim(refined_img, gt_img)
        psnr_val = compute_psnr(refined_img, gt_img)
        
        # Classification
        pred_class, pred_probs = classify_image_clip(refined_img, clip_model, clip_processor, device)
        
        if pred_class == concept_id:
            correct_40way += 1
        
        top5_preds = np.argsort(pred_probs)[-5:]
        if concept_id in top5_preds:
            correct_top5 += 1
        
        gt_is_animal = concept_id < 10
        pred_is_animal = pred_class < 10
        if gt_is_animal == pred_is_animal:
            correct_2way += 1
        
        total_samples += 1
        
        result = {
            'sample_idx': batch_idx,
            'subject_id': subject_id,
            'concept_id': concept_id,
            'concept_name': SEED_DV_CONCEPTS[concept_id],
            'pred_class': pred_class,
            'pred_name': SEED_DV_CONCEPTS[pred_class],
            'ssim': float(ssim_val),
            'psnr': float(psnr_val),
            'correct_40way': bool(pred_class == concept_id),
            'correct_top5': bool(concept_id in top5_preds),
            'correct_2way': bool(gt_is_animal == pred_is_animal),
        }
        all_results.append(result)
        
        # Save images
        if batch_idx < 100 or batch_idx % 50 == 0:
            Image.fromarray(refined_img).save(output_dir / "predictions" / f"{batch_idx:05d}.png")
            Image.fromarray(gt_img).save(output_dir / "ground_truth" / f"{batch_idx:05d}.png")
            Image.fromarray(blurry_img).save(output_dir / "blurry" / f"{batch_idx:05d}.png")
            # Comparison: GT | Blurry | Refined
            comparison = np.hstack([gt_img, blurry_img, refined_img])
            Image.fromarray(comparison).save(output_dir / "comparisons" / f"{batch_idx:05d}.png")
    
    # Summary
    ssim_values = [r['ssim'] for r in all_results]
    psnr_values = [r['psnr'] for r in all_results]
    
    summary = {
        'total_samples': total_samples,
        'settings': {
            'strength': args.strength,
            'num_inference_steps': args.num_inference_steps,
            'guidance_scale': args.guidance_scale,
            'use_captions': use_captions,
        },
        'metrics': {
            'ssim_mean': float(np.mean(ssim_values)),
            'ssim_std': float(np.std(ssim_values)),
            'psnr_mean': float(np.mean(psnr_values)),
            'psnr_std': float(np.std(psnr_values)),
            'acc_40way': float(correct_40way / total_samples),
            'acc_top5': float(correct_top5 / total_samples),
            'acc_2way': float(correct_2way / total_samples),
        },
    }
    
    # Print results
    caption_mode = "WITH captions" if use_captions else "WITHOUT captions"
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS ({caption_mode} + Diffusion)")
    print("="*60)
    print(f"\nSettings: strength={args.strength}, steps={args.num_inference_steps}")
    print(f"Total samples: {total_samples}")
    print(f"\nPixel-level metrics:")
    print(f"  SSIM: {summary['metrics']['ssim_mean']:.3f} ± {summary['metrics']['ssim_std']:.3f}")
    print(f"  PSNR: {summary['metrics']['psnr_mean']:.2f} ± {summary['metrics']['psnr_std']:.2f}")
    print(f"\nSemantic-level metrics:")
    print(f"  40-way accuracy: {summary['metrics']['acc_40way']*100:.1f}%")
    print(f"  Top-5 accuracy: {summary['metrics']['acc_top5']*100:.1f}%")
    print(f"  2-way accuracy: {summary['metrics']['acc_2way']*100:.1f}%")
    
    print("\n" + "="*60)
    print("COMPARISON WITH EEG2Video (NeurIPS 2024)")
    print("="*60)
    print(f"{'Metric':<20} {'EEG2Video':<15} {'Ours (Diffusion)':<20}")
    print("-"*55)
    print(f"{'SSIM':<20} {'0.256 ± 0.03':<15} {summary['metrics']['ssim_mean']:.3f} ± {summary['metrics']['ssim_std']:.2f}")
    print(f"{'40-way acc':<20} {'15.9%':<15} {summary['metrics']['acc_40way']*100:.1f}%")
    print(f"{'2-way acc':<20} {'79.8%':<15} {summary['metrics']['acc_2way']*100:.1f}%")
    
    # Save results
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
