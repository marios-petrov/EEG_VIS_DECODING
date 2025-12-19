#!/usr/bin/env python3
"""
Evaluate EEGMamba + SD 3.5 on SEED-DV dataset (Direct Latent Decode).

Computes same metrics as EEG2Video paper (NeurIPS 2024):
- SSIM: Structural similarity between generated and ground truth frames
- N-way semantic accuracy: Using CLIP classifier

This version uses direct VAE decoding (faster but blurrier).
For diffusion-refined evaluation, use eval_eegmamba_seed_dv_diffusion.py

Usage:
    python eval_eegmamba_seed_dv.py \
        --checkpoint /path/to/adapter_best.pt \
        --preprocessed-dir /path/to/preprocessed \
        --output-dir /path/to/eval_results
"""

import os
os.environ['HF_HOME'] = '/local-scratch/marios-datasets/hf-cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/local-scratch/marios-datasets/hf-cache/hub'

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import cv2
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Import EEGMamba adapter
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
        self.frame_data = {}
        
        for subj_id in subject_ids:
            # Try multiple naming conventions
            subj_dir = self.preprocessed_dir / f"sub{subj_id}"
            if not subj_dir.exists():
                subj_dir = self.preprocessed_dir / f"subject_{subj_id:02d}"
            if not subj_dir.exists():
                subj_dir = self.preprocessed_dir / f"subject_{subj_id}"
            if not subj_dir.exists():
                print(f"Warning: Subject {subj_id} not found")
                continue
            
            # Load data - try different file naming conventions
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
            
            # Load frames if available
            frames_path = subj_dir / "frames.npy"
            if not frames_path.exists():
                frames_path = subj_dir / "video_frames.npy"
            if frames_path.exists():
                self.frame_data[subj_id] = np.load(frames_path)
            
            for seg in metadata['segments']:
                if not seg['is_train']:  # Test only
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
        target_latent = latent[len(latent)//2]  # Middle frame
        
        # Get caption embedding
        if self.precomputed_embeddings is not None:
            caption = seg['caption']
            caption_emb = self.precomputed_embeddings.get(
                caption, torch.zeros(2048)
            )
        else:
            caption_emb = torch.zeros(2048)
        
        return {
            'eeg': torch.from_numpy(eeg).float(),
            'latent': torch.from_numpy(target_latent).float(),
            'all_latents': torch.from_numpy(latent).float(),
            'caption_emb': caption_emb,
            'subject_id': subj_id,
            'concept_id': seg['concept_id'],
            'seg_idx': seg_idx,
            'caption': seg['caption'],
        }


def load_vae():
    """Load SD 3.5 VAE for decoding latents"""
    from diffusers import AutoencoderKL
    
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    vae.eval()
    return vae


def decode_latent(latent: torch.Tensor, vae, device) -> np.ndarray:
    """Decode a single latent to image"""
    if latent.dim() == 3:
        latent = latent.unsqueeze(0)
    
    latent = latent.to(device, dtype=torch.float16)
    
    # SD 3.5 uses different scaling
    latent = (latent / vae.config.scaling_factor) + vae.config.shift_factor
    
    with torch.no_grad():
        image = vae.decode(latent, return_dict=False)[0]
    
    # Convert to numpy
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    
    return image[0]


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two images"""
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
    
    return ssim(img1_gray, img2_gray, data_range=255)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images"""
    return psnr(img1, img2, data_range=255)


def load_clip_classifier(device):
    """Load CLIP model for semantic classification"""
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
    """Classify image using CLIP zero-shot classification"""
    
    # Prepare image
    pil_image = Image.fromarray(image)
    
    # Prepare text prompts
    text_prompts = [f"a video frame of {concept}" for concept in concepts]
    
    # Process inputs
    inputs = processor(
        text=text_prompts,
        images=pil_image,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]
        probs = torch.softmax(logits, dim=0).cpu().numpy()
    
    predicted_class = int(np.argmax(probs))
    return predicted_class, probs


def evaluate(
    model,
    vae,
    dataloader,
    clip_model,
    clip_processor,
    device,
    output_dir: Path,
    max_samples: int = None,
    use_captions: bool = True,
):
    """Run full evaluation
    
    Args:
        use_captions: If False, zero out caption embeddings (ablation study)
    """
    model.eval()
    
    all_results = []
    per_concept_metrics = defaultdict(lambda: defaultdict(list))
    per_subject_metrics = defaultdict(lambda: defaultdict(list))
    
    # For N-way accuracy
    correct_40way = 0
    correct_2way = 0  # Simplified 2-way (animal vs non-animal or similar)
    total_samples = 0
    
    # For top-5 accuracy
    correct_top5 = 0
    
    # Create output directories
    (output_dir / "predictions").mkdir(exist_ok=True, parents=True)
    (output_dir / "ground_truth").mkdir(exist_ok=True)
    (output_dir / "comparisons").mkdir(exist_ok=True)
    
    print(f"Running evaluation (captions: {'enabled' if use_captions else 'DISABLED'})...")
    
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        if max_samples and batch_idx * dataloader.batch_size >= max_samples:
            break
        
        eeg = batch['eeg'].to(device)
        target_latent = batch['latent'].to(device)
        caption_emb = batch['caption_emb'].to(device)
        concept_ids = batch['concept_id'].numpy()
        subject_ids = batch['subject_id'].numpy()
        
        # Zero out captions if ablation mode
        if not use_captions:
            caption_emb = torch.zeros_like(caption_emb)
        
        # Generate prediction
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred_latent = model(eeg, caption_emb)
        
        # Process each sample in batch
        for i in range(len(eeg)):
            sample_idx = batch_idx * dataloader.batch_size + i
            
            # Decode latents to images
            pred_img = decode_latent(pred_latent[i], vae, device)
            gt_img = decode_latent(target_latent[i], vae, device)
            
            # Compute metrics
            ssim_val = compute_ssim(pred_img, gt_img)
            psnr_val = compute_psnr(pred_img, gt_img)
            
            # Semantic classification
            pred_class, pred_probs = classify_image_clip(
                pred_img, clip_model, clip_processor, device
            )
            gt_class = concept_ids[i]
            
            # 40-way accuracy
            if pred_class == gt_class:
                correct_40way += 1
            
            # Top-5 accuracy
            top5_preds = np.argsort(pred_probs)[-5:]
            if gt_class in top5_preds:
                correct_top5 += 1
            
            # 2-way accuracy (simplified: first 10 concepts vs rest)
            # This mimics animal vs non-animal distinction
            gt_is_animal = gt_class < 10
            pred_is_animal = pred_class < 10
            is_correct_2way = (gt_is_animal == pred_is_animal)
            if is_correct_2way:
                correct_2way += 1
            
            total_samples += 1
            
            # Store results - FIXED: explicit bool() conversion for JSON serialization
            result = {
                'sample_idx': sample_idx,
                'subject_id': int(subject_ids[i]),
                'concept_id': int(gt_class),
                'concept_name': SEED_DV_CONCEPTS[gt_class],
                'pred_class': int(pred_class),
                'pred_name': SEED_DV_CONCEPTS[pred_class],
                'ssim': float(ssim_val),
                'psnr': float(psnr_val),
                'correct_40way': bool(pred_class == gt_class),
                'correct_top5': bool(gt_class in top5_preds),
                'correct_2way': bool(is_correct_2way),
            }
            all_results.append(result)
            
            # Per-concept metrics
            per_concept_metrics[gt_class]['ssim'].append(ssim_val)
            per_concept_metrics[gt_class]['psnr'].append(psnr_val)
            per_concept_metrics[gt_class]['correct'].append(pred_class == gt_class)
            
            # Per-subject metrics
            per_subject_metrics[subject_ids[i]]['ssim'].append(ssim_val)
            per_subject_metrics[subject_ids[i]]['psnr'].append(psnr_val)
            per_subject_metrics[subject_ids[i]]['correct'].append(pred_class == gt_class)
            
            # Save some example images
            if sample_idx < 100 or sample_idx % 100 == 0:
                # Save prediction
                Image.fromarray(pred_img).save(
                    output_dir / "predictions" / f"{sample_idx:05d}.png"
                )
                # Save ground truth
                Image.fromarray(gt_img).save(
                    output_dir / "ground_truth" / f"{sample_idx:05d}.png"
                )
                # Save comparison
                comparison = np.hstack([gt_img, pred_img])
                Image.fromarray(comparison).save(
                    output_dir / "comparisons" / f"{sample_idx:05d}.png"
                )
    
    # Compute summary statistics
    ssim_values = [r['ssim'] for r in all_results]
    psnr_values = [r['psnr'] for r in all_results]
    
    summary = {
        'total_samples': total_samples,
        'metrics': {
            'ssim_mean': float(np.mean(ssim_values)),
            'ssim_std': float(np.std(ssim_values)),
            'psnr_mean': float(np.mean(psnr_values)),
            'psnr_std': float(np.std(psnr_values)),
            'acc_40way': float(correct_40way / total_samples),
            'acc_top5': float(correct_top5 / total_samples),
            'acc_2way': float(correct_2way / total_samples),
        },
        'per_concept': {},
        'per_subject': {},
    }
    
    # Per-concept summary
    for concept_id, metrics in per_concept_metrics.items():
        summary['per_concept'][SEED_DV_CONCEPTS[concept_id]] = {
            'ssim_mean': float(np.mean(metrics['ssim'])),
            'psnr_mean': float(np.mean(metrics['psnr'])),
            'accuracy': float(np.mean(metrics['correct'])),
            'n_samples': len(metrics['ssim']),
        }
    
    # Per-subject summary
    for subj_id, metrics in per_subject_metrics.items():
        summary['per_subject'][int(subj_id)] = {
            'ssim_mean': float(np.mean(metrics['ssim'])),
            'psnr_mean': float(np.mean(metrics['psnr'])),
            'accuracy': float(np.mean(metrics['correct'])),
            'n_samples': len(metrics['ssim']),
        }
    
    return summary, all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--preprocessed-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--subjects', type=str, default='1-20')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no-captions', action='store_true',
                       help='Evaluate without text conditioning (ablation study)')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse subjects
    if '-' in args.subjects:
        start, end = map(int, args.subjects.split('-'))
        subject_ids = list(range(start, end + 1))
    else:
        subject_ids = [int(s) for s in args.subjects.split(',')]
    
    print(f"Evaluating on subjects: {subject_ids}")
    
    # Load models
    print("\nLoading models...")
    
    # Load VAE
    vae = load_vae().to(device)
    
    # Load CLIP
    clip_model, clip_processor = load_clip_classifier(device)
    
    # Load EEGMamba adapter
    from eegmamba_adapter_optimized import EEGMambaAdapter
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
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
    
    # Load caption embeddings if available
    # Try checkpoint directory first, then preprocessed parent
    checkpoint_dir = Path(args.checkpoint).parent
    embeddings_path = checkpoint_dir / "caption_embeddings.pt"
    if not embeddings_path.exists():
        embeddings_path = Path(args.preprocessed_dir).parent / "caption_embeddings.pt"
    
    caption_embeddings = None
    if embeddings_path.exists():
        caption_embeddings = torch.load(embeddings_path)
        print(f"Loaded {len(caption_embeddings)} caption embeddings from {embeddings_path}")
    
    # Create test dataset
    test_dataset = SEEDDVTestDataset(
        Path(args.preprocessed_dir),
        subject_ids,
        precomputed_embeddings=caption_embeddings,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Determine output directory based on caption mode
    use_captions = not args.no_captions
    if args.no_captions:
        output_dir = output_dir / "no_captions"
        print("\n⚠️  ABLATION MODE: Evaluating WITHOUT captions")
    else:
        output_dir = output_dir / "with_captions"
        print("\n✓ Evaluating WITH captions")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    summary, all_results = evaluate(
        model, vae, test_loader, clip_model, clip_processor,
        device, output_dir, max_samples=args.max_samples,
        use_captions=use_captions
    )
    
    # Add caption mode to summary
    summary['use_captions'] = use_captions
    
    # Print results
    caption_mode = "WITH captions" if use_captions else "WITHOUT captions"
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS ({caption_mode})")
    print("="*60)
    print(f"\nTotal samples: {summary['total_samples']}")
    print(f"\nPixel-level metrics:")
    print(f"  SSIM: {summary['metrics']['ssim_mean']:.3f} ± {summary['metrics']['ssim_std']:.3f}")
    print(f"  PSNR: {summary['metrics']['psnr_mean']:.2f} ± {summary['metrics']['psnr_std']:.2f}")
    print(f"\nSemantic-level metrics:")
    print(f"  40-way accuracy: {summary['metrics']['acc_40way']*100:.1f}%")
    print(f"  Top-5 accuracy: {summary['metrics']['acc_top5']*100:.1f}%")
    print(f"  2-way accuracy: {summary['metrics']['acc_2way']*100:.1f}%")
    
    # Compare with EEG2Video
    print("\n" + "="*60)
    print("COMPARISON WITH EEG2Video (NeurIPS 2024)")
    print("="*60)
    print(f"{'Metric':<20} {'EEG2Video':<15} {'Ours (EEGMamba+SD3.5)':<20}")
    print("-"*55)
    print(f"{'SSIM':<20} {'0.256 ± 0.03':<15} {summary['metrics']['ssim_mean']:.3f} ± {summary['metrics']['ssim_std']:.2f}")
    print(f"{'40-way acc':<20} {'15.9%':<15} {summary['metrics']['acc_40way']*100:.1f}%")
    print(f"{'2-way acc':<20} {'79.8%':<15} {summary['metrics']['acc_2way']*100:.1f}%")
    
    # Save results
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save CSVs for easy analysis
    import csv
    
    # Per-sample CSV
    with open(output_dir / "per_sample_metrics.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'sample_idx', 'subject_id', 'concept_id', 'concept_name',
            'ssim', 'psnr', 'pred_class', 'correct_40way', 'correct_2way'
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                'sample_idx': r['sample_idx'],
                'subject_id': r['subject_id'],
                'concept_id': r['concept_id'],
                'concept_name': r['concept_name'],
                'ssim': r['ssim'],
                'psnr': r['psnr'],
                'pred_class': r['pred_class'],
                'correct_40way': int(r['correct_40way']),
                'correct_2way': int(r['correct_2way']),
            })
    
    # Per-subject CSV
    with open(output_dir / "per_subject_summary.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subject_id', 'n_samples', 'ssim_mean', 'psnr_mean', 'accuracy_40way'])
        for subj_id, metrics in sorted(summary['per_subject'].items()):
            writer.writerow([
                subj_id, metrics['n_samples'], 
                f"{metrics['ssim_mean']:.4f}",
                f"{metrics['psnr_mean']:.2f}",
                f"{metrics['accuracy']*100:.1f}%"
            ])
    
    # Per-concept CSV
    with open(output_dir / "per_concept_summary.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['concept_id', 'concept_name', 'n_samples', 'ssim_mean', 'accuracy_40way'])
        for i, concept in enumerate(SEED_DV_CONCEPTS):
            if concept in summary['per_concept']:
                m = summary['per_concept'][concept]
                writer.writerow([
                    i, concept, m['n_samples'],
                    f"{m['ssim_mean']:.4f}",
                    f"{m['accuracy']*100:.1f}%"
                ])
    
    # Paper comparison CSV
    with open(output_dir / "paper_comparison.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'EEG2Video (Paper)', 'Ours (EEGMamba+SD3.5)', 'Difference'])
        
        our_ssim = summary['metrics']['ssim_mean']
        our_40way = summary['metrics']['acc_40way'] * 100
        our_2way = summary['metrics']['acc_2way'] * 100
        
        writer.writerow(['SSIM', '0.256', f"{our_ssim:.3f}", f"{our_ssim - 0.256:+.3f}"])
        writer.writerow(['40-way accuracy (%)', '15.9', f"{our_40way:.1f}", f"{our_40way - 15.9:+.1f}"])
        writer.writerow(['2-way accuracy (%)', '79.8', f"{our_2way:.1f}", f"{our_2way - 79.8:+.1f}"])
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - summary.json")
    print(f"  - all_results.json")
    print(f"  - per_sample_metrics.csv")
    print(f"  - per_subject_summary.csv")
    print(f"  - per_concept_summary.csv")
    print(f"  - paper_comparison.csv")


if __name__ == "__main__":
    main()
