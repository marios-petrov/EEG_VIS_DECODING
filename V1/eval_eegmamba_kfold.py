#!/usr/bin/env python3
"""
Evaluate EEGMamba Adapter with K-Fold Cross-Validation

Evaluates ONLY on held-out test subjects from each fold.

Usage:
    # Evaluate single fold
    python eval_eegmamba_kfold.py \
        --checkpoint /path/to/fold0/adapter_best.pt \
        --preprocessed-dir /path/to/data \
        --output-dir /path/to/eval/fold0 \
        --task dme

    # After running all 5 folds, aggregate results:
    python eval_eegmamba_kfold.py --aggregate-results /path/to/eval
"""

import subprocess
import sys
try:
    import cv2
except ImportError:
    print("📦 Installing OpenCV dependencies...")
    subprocess.run(["apt-get", "update"], check=False, capture_output=True)
    subprocess.run(["apt-get", "install", "-y", "libgl1", "libglib2.0-0"], check=False, capture_output=True)
    import cv2

import os
os.environ['HF_HOME'] = '/local-scratch/marios-datasets/hf-cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/local-scratch/marios-datasets/hf-cache/hub'

sys.path.insert(0, '/local-scratch/marios-datasets/EEGMamba')

import argparse
import json
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image, ImageDraw
import cv2

try:
    import lpips
    LPIPS_AVAILABLE = True
except:
    LPIPS_AVAILABLE = False

try:
    from transformers import CLIPModel, CLIPProcessor
    CLIP_AVAILABLE = True
except:
    CLIP_AVAILABLE = False

from eegmamba_adapter_optimized import EEGMambaAdapter


def get_fold_subjects(fold: int, num_folds: int = 5, num_subjects: int = 22):
    """Get train and test subject IDs for a given fold."""
    fold_test_subjects = {
        0: list(range(0, 5)),
        1: list(range(5, 10)),
        2: list(range(10, 14)),
        3: list(range(14, 18)),
        4: list(range(18, 22)),
    }
    test_subjects = set(fold_test_subjects[fold])
    train_subjects = set(range(num_subjects)) - test_subjects
    return sorted(list(train_subjects)), sorted(list(test_subjects))


class EEGVideoDatasetKFold(Dataset):
    """Dataset filtered to specific subjects"""
    def __init__(self, preprocessed_dir: Path, task: str, subject_ids: list, num_total_subjects: int = 22):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.subject_ids = set(subject_ids)
        self.num_total_subjects = num_total_subjects
        
        self.eeg_mmap = np.load(preprocessed_dir / f"{task}_eeg.npy", mmap_mode='r')
        self.latents_mmap = np.load(preprocessed_dir / f"{task}_vae_latents_hd.npy", mmap_mode='r')
        
        captions_path = preprocessed_dir / f"{task}_captions_hd.json"
        if captions_path.exists():
            with open(captions_path) as f:
                data = json.load(f)
                self.captions = data.get('captions', [])
        else:
            self.captions = []
        
        self.num_segments = self.eeg_mmap.shape[0]
        self.frames_per_segment = self.latents_mmap.shape[1]
        
        # Build valid indices (only test subjects)
        self.valid_indices = []
        for segment_idx in range(self.num_segments):
            subject_id = segment_idx % num_total_subjects
            if subject_id in self.subject_ids:
                for frame_idx in range(self.frames_per_segment):
                    global_idx = segment_idx * self.frames_per_segment + frame_idx
                    self.valid_indices.append(global_idx)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        seg_idx = actual_idx // self.frames_per_segment
        frame_idx = actual_idx % self.frames_per_segment
        
        return {
            'eeg': torch.from_numpy(self.eeg_mmap[seg_idx].copy()).float(),
            'latent': torch.from_numpy(self.latents_mmap[seg_idx, frame_idx].copy()).float(),
            'caption': self.captions[seg_idx] if seg_idx < len(self.captions) else "",
            'subject_id': torch.tensor(seg_idx % self.num_total_subjects, dtype=torch.long),
            'seg_idx': seg_idx,
            'frame_idx': frame_idx,
        }


def calc_ssim(img1, img2, window_size=11):
    C1, C2 = 0.01**2, 0.03**2
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=img1.device)
    coords -= window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = g.outer(g).unsqueeze(0).unsqueeze(0).expand(3, 1, window_size, window_size)
    pad = window_size // 2
    
    mu1 = F.conv2d(img1, window, padding=pad, groups=3)
    mu2 = F.conv2d(img2, window, padding=pad, groups=3)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = F.conv2d(img1**2, window, padding=pad, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(img2**2, window, padding=pad, groups=3) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=pad, groups=3) - mu1_mu2
    
    ssim = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim.mean().item()


def calc_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    return 100.0 if mse == 0 else (10 * torch.log10(1.0 / mse)).item()


def tensor_to_pil(t):
    img = t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))


def create_comparison(gt_pil, pred_pil, metrics, idx, subject_id):
    w, h = gt_pil.size
    canvas = Image.new('RGB', (w*2 + 20, h + 50), (255, 255, 255))
    canvas.paste(gt_pil, (0, 0))
    canvas.paste(pred_pil, (w + 20, 0))
    
    draw = ImageDraw.Draw(canvas)
    draw.text((w//2 - 30, h + 5), "Ground Truth", fill=(0, 0, 0))
    draw.text((w + 20 + w//2 - 25, h + 5), "Predicted", fill=(0, 0, 0))
    
    m_str = f"#{idx} Subj:{subject_id}  SSIM:{metrics['ssim']:.3f}  PSNR:{metrics['psnr']:.1f}dB"
    if 'lpips' in metrics:
        m_str += f"  LPIPS:{metrics['lpips']:.3f}"
    if 'clip' in metrics:
        m_str += f"  CLIP:{metrics['clip']:.3f}"
    draw.text((10, h + 28), m_str, fill=(80, 80, 80))
    
    return canvas


def aggregate_fold_results(base_dir: Path):
    """Aggregate results from all folds into a single summary"""
    print("="*70)
    print("AGGREGATING K-FOLD RESULTS")
    print("="*70)
    
    all_fold_results = []
    all_subject_metrics = defaultdict(lambda: defaultdict(list))
    
    for fold in range(5):
        fold_dir = base_dir / f"fold{fold}"
        results_path = fold_dir / "results.json"
        
        if not results_path.exists():
            print(f"⚠ Fold {fold} results not found at {results_path}")
            continue
        
        with open(results_path) as f:
            results = json.load(f)
        
        all_fold_results.append({
            'fold': fold,
            'test_subjects': results['config'].get('test_subjects', []),
            'overall': results['overall'],
            'per_subject': results['per_subject'],
        })
        
        # Collect per-subject metrics
        for subj_id, subj_data in results['per_subject'].items():
            for metric, vals in subj_data.items():
                if metric != 'count':
                    all_subject_metrics[int(subj_id)][metric].append(vals['mean'])
    
    if not all_fold_results:
        print("❌ No fold results found!")
        return
    
    # Compute overall metrics across all folds
    print(f"\n✓ Found {len(all_fold_results)} fold results")
    
    # Aggregate
    all_metrics = defaultdict(list)
    for fold_result in all_fold_results:
        for metric, vals in fold_result['overall'].items():
            all_metrics[metric].append(vals['mean'])
    
    print("\n" + "="*70)
    print("CROSS-VALIDATED RESULTS (HELD-OUT SUBJECTS)")
    print("="*70)
    
    summary = {}
    for metric, values in all_metrics.items():
        summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'per_fold': values,
        }
        arrow = "↑" if metric in ['ssim', 'psnr', 'clip'] else "↓"
        print(f"  {metric.upper():8s}: {summary[metric]['mean']:.4f} ± {summary[metric]['std']:.4f} {arrow}")
    
    # Per-fold breakdown
    print("\n📊 Per-Fold Results:")
    for fold_result in all_fold_results:
        fold = fold_result['fold']
        test_subjs = fold_result['test_subjects']
        ssim = fold_result['overall'].get('ssim', {}).get('mean', 0)
        psnr = fold_result['overall'].get('psnr', {}).get('mean', 0)
        clip_score = fold_result['overall'].get('clip', {}).get('mean', 0)
        print(f"  Fold {fold}: SSIM={ssim:.4f}, PSNR={psnr:.2f}, CLIP={clip_score:.4f} | Test subjects: {test_subjs}")
    
    # Per-subject summary
    print("\n📊 Per-Subject Results (across folds):")
    subject_summary = {}
    for subj_id in sorted(all_subject_metrics.keys()):
        subj_data = all_subject_metrics[subj_id]
        subject_summary[subj_id] = {}
        ssim_mean = np.mean(subj_data.get('ssim', [0]))
        psnr_mean = np.mean(subj_data.get('psnr', [0]))
        clip_mean = np.mean(subj_data.get('clip', [0]))
        print(f"  Subject {subj_id:2d}: SSIM={ssim_mean:.4f}, PSNR={psnr_mean:.2f}, CLIP={clip_mean:.4f}")
        subject_summary[subj_id] = {'ssim': ssim_mean, 'psnr': psnr_mean, 'clip': clip_mean}
    
    # Save aggregated results
    agg_path = base_dir / "aggregated_results.json"
    with open(agg_path, 'w') as f:
        json.dump({
            'overall': {k: {kk: float(vv) if not isinstance(vv, list) else vv 
                           for kk, vv in v.items()} for k, v in summary.items()},
            'per_fold': all_fold_results,
            'per_subject': {str(k): v for k, v in subject_summary.items()},
        }, f, indent=2)
    
    print(f"\n💾 Aggregated results saved to: {agg_path}")
    print("\n" + "="*70)
    print("✅ AGGREGATION COMPLETE")
    print("="*70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='Adapter checkpoint path')
    parser.add_argument('--preprocessed-dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--task', default='dme', choices=['dme', 'tp', 'inscapes'])
    parser.add_argument('--num-samples', type=int, default=999999, help='Max samples to evaluate')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--skip-video', action='store_true')
    parser.add_argument('--skip-clip', action='store_true')
    parser.add_argument('--aggregate-results', type=str, help='Aggregate results from all folds in this directory')
    args = parser.parse_args()
    
    # Aggregation mode
    if args.aggregate_results:
        aggregate_fold_results(Path(args.aggregate_results))
        return
    
    # Evaluation mode
    if not args.checkpoint or not args.preprocessed_dir or not args.output_dir:
        parser.error("--checkpoint, --preprocessed-dir, and --output-dir are required for evaluation")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    (output_dir / "ground_truth").mkdir(exist_ok=True)
    
    print("="*70)
    print("EEGMAMBA + SD 3.5 EVALUATION (K-FOLD)")
    print("="*70)
    
    # Load checkpoint to get fold info
    print(f"\n🧠 Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    fold = ckpt.get('fold', None)
    test_subjects = ckpt.get('test_subjects', None)
    
    if test_subjects is None:
        print("⚠ No fold info in checkpoint, inferring from filename...")
        # Try to infer from path
        ckpt_path = Path(args.checkpoint)
        for part in ckpt_path.parts:
            if 'fold' in part.lower():
                try:
                    fold = int(part.replace('fold', ''))
                    _, test_subjects = get_fold_subjects(fold)
                    break
                except:
                    pass
        
        if test_subjects is None:
            print("❌ Cannot determine test subjects. Please ensure checkpoint has fold info.")
            return
    
    print(f"  Fold: {fold}")
    print(f"  Test subjects: {test_subjects}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    print(f"  Training loss: {ckpt.get('loss', '?'):.4f}")
    
    # Load dataset (test subjects only)
    print(f"\n📂 Loading dataset (test subjects only)...")
    dataset = EEGVideoDatasetKFold(
        Path(args.preprocessed_dir), args.task, test_subjects
    )
    num_samples = min(args.num_samples, len(dataset))
    print(f"  Test samples: {len(dataset):,}")
    print(f"  Evaluating: {num_samples:,}")
    
    # Load VAE
    print("\n🎨 Loading SD 3.5 VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(device)
    vae.eval()
    print("  ✓ VAE loaded")
    
    # Load adapter
    sample = dataset[0]
    in_channels = sample['eeg'].shape[0]
    eeg_time_steps = sample['eeg'].shape[1]
    
    adapter = EEGMambaAdapter(
        in_channels=in_channels,
        eeg_time_steps=eeg_time_steps,
        latent_channels=16,
        latent_height=96,
        latent_width=96,
        use_glmnet=True,
        num_subjects=22,
        freeze_eegmamba=True,
    ).to(device)
    
    adapter.load_state_dict(ckpt['adapter'])
    adapter.eval()
    print("  ✓ Adapter loaded")
    
    # Load metrics
    lpips_model = None
    if LPIPS_AVAILABLE:
        print("\n📏 Loading LPIPS...")
        lpips_model = lpips.LPIPS(net='alex').to(device)
    
    clip_model, clip_processor = None, None
    if CLIP_AVAILABLE and not args.skip_clip:
        print("📏 Loading CLIP...")
        try:
            clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32", use_safetensors=True
            ).to(device)
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("  ✓ CLIP loaded")
        except Exception as e:
            print(f"  ⚠ CLIP failed: {e}")
    
    # Evaluate
    print("\n" + "="*70)
    print(f"EVALUATING ON HELD-OUT SUBJECTS: {test_subjects}")
    print("="*70)
    
    all_metrics = defaultdict(list)
    per_subject_metrics = defaultdict(lambda: defaultdict(list))
    per_subject_frames = defaultdict(lambda: {'pred': [], 'gt': []})
    results = []
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Evaluating"):
            sample = dataset[i]
            
            eeg = sample['eeg'].unsqueeze(0).to(device)
            subject_id_tensor = sample['subject_id'].unsqueeze(0).to(device)
            gt_latent = sample['latent'].unsqueeze(0).to(device)
            
            pred_latent = adapter(eeg, subject_id_tensor)
            
            pred_frame = vae.decode(pred_latent / vae.config.scaling_factor).sample
            gt_frame = vae.decode(gt_latent / vae.config.scaling_factor).sample
            
            pred_frame = ((pred_frame + 1) / 2).clamp(0, 1)
            gt_frame = ((gt_frame + 1) / 2).clamp(0, 1)
            
            metrics = {
                'ssim': calc_ssim(pred_frame, gt_frame),
                'psnr': calc_psnr(pred_frame, gt_frame),
            }
            
            if lpips_model:
                metrics['lpips'] = lpips_model(pred_frame * 2 - 1, gt_frame * 2 - 1).item()
            
            pred_pil = tensor_to_pil(pred_frame)
            gt_pil = tensor_to_pil(gt_frame)
            
            if clip_model:
                inputs = clip_processor(images=[pred_pil, gt_pil], return_tensors="pt").to(device)
                feats = clip_model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                metrics['clip'] = (feats[0] @ feats[1]).item()
            
            subject_id_val = sample['subject_id'].item()
            for k, v in metrics.items():
                all_metrics[k].append(v)
                per_subject_metrics[subject_id_val][k].append(v)
            
            results.append({
                'idx': i, 'seg_idx': sample['seg_idx'], 'frame_idx': sample['frame_idx'],
                'subject_id': subject_id_val, **metrics
            })
            
            # Save individual images
            pred_pil.save(output_dir / "predictions" / f"{i:05d}.png")
            gt_pil.save(output_dir / "ground_truth" / f"{i:05d}.png")
            
            # Collect frames for video (only if not skipping video)
            if not args.skip_video:
                per_subject_frames[subject_id_val]['pred'].append(np.array(pred_pil))
                per_subject_frames[subject_id_val]['gt'].append(np.array(gt_pil))
    
    # Create videos (predictions only, organized by subject)
    if not args.skip_video:
        print("\n🎬 Creating per-subject videos...")
        (output_dir / "videos").mkdir(exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        for subj_id in sorted(per_subject_frames.keys()):
            pred_frames = per_subject_frames[subj_id]['pred']
            gt_frames = per_subject_frames[subj_id]['gt']
            
            if pred_frames:
                h, w = pred_frames[0].shape[:2]
                
                # Prediction video
                pred_path = output_dir / "videos" / f"subject_{subj_id:02d}_predicted.mp4"
                out = cv2.VideoWriter(str(pred_path), fourcc, 6, (w, h))
                for frame in pred_frames:
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                out.release()
                
                # Ground truth video
                gt_path = output_dir / "videos" / f"subject_{subj_id:02d}_groundtruth.mp4"
                out = cv2.VideoWriter(str(gt_path), fourcc, 6, (w, h))
                for frame in gt_frames:
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                out.release()
                
                print(f"  ✓ Subject {subj_id}: {len(pred_frames)} frames")
    
    # Summary
    print("\n" + "="*70)
    print(f"FOLD {fold} RESULTS (HELD-OUT SUBJECTS: {test_subjects})")
    print("="*70)
    
    summary = {}
    for k, v in all_metrics.items():
        summary[k] = {'mean': np.mean(v), 'std': np.std(v), 'min': np.min(v), 'max': np.max(v)}
        arrow = "↑" if k in ['ssim', 'psnr', 'clip'] else "↓"
        print(f"  {k.upper():8s}: {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f} {arrow}")
    
    print("\n📊 Per-Subject Results:")
    subject_summary = {}
    for subj_id in sorted(per_subject_metrics.keys()):
        subject_summary[int(subj_id)] = {}
        print(f"  Subject {subj_id}:")
        for k, v in per_subject_metrics[subj_id].items():
            subject_summary[int(subj_id)][k] = {
                'mean': float(np.mean(v)), 'std': float(np.std(v)), 'count': len(v)
            }
            arrow = "↑" if k in ['ssim', 'psnr', 'clip'] else "↓"
            print(f"    {k.upper():8s}: {np.mean(v):.4f} ± {np.std(v):.4f} (n={len(v)}) {arrow}")
    
    # Save results
    with open(output_dir / "results.json", 'w') as f:
        json.dump({
            'fold': fold,
            'test_subjects': test_subjects,
            'overall': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in summary.items()},
            'per_subject': subject_summary,
            'per_sample': results,
            'config': {**vars(args), 'test_subjects': test_subjects},
        }, f, indent=2)
    
    with open(output_dir / "metrics.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    
    print(f"\n💾 Results saved to: {output_dir}")
    print("\n✅ EVALUATION COMPLETE!")


if __name__ == "__main__":
    main()
