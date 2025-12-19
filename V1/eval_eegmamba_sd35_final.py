#!/usr/bin/env python3
"""
Quick Evaluation for EEGMamba Adapter + SD 3.5
==============================================
Generates side-by-side comparisons and computes metrics.

Usage:
    python eval_quick.py \
        --checkpoint /path/to/adapter_epoch100.pt \
        --preprocessed-dir /path/to/preprocessed_dme_sd35 \
        --output-dir /path/to/eval_output \
        --task dme \
        --num-samples 100
"""

# Install OpenCV dependencies if needed
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

import sys
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
from PIL import Image, ImageDraw, ImageFont
import cv2

# Metrics
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

# FIXED: Import from optimized adapter (same as training script)
from eegmamba_adapter_optimized import EEGMambaAdapter


class EEGVideoDataset(Dataset):
    def __init__(self, preprocessed_dir: Path, task: str):
        self.preprocessed_dir = Path(preprocessed_dir)
        
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
    
    def __len__(self):
        return self.num_segments * self.frames_per_segment
    
    def __getitem__(self, idx):
        seg_idx = idx // self.frames_per_segment
        frame_idx = idx % self.frames_per_segment

        return {
            'eeg': torch.from_numpy(self.eeg_mmap[seg_idx].copy()).float(),
            'latent': torch.from_numpy(self.latents_mmap[seg_idx, frame_idx].copy()).float(),
            'caption': self.captions[seg_idx] if seg_idx < len(self.captions) else "",
            'subject_id': torch.tensor(seg_idx % 22, dtype=torch.long),
            'seg_idx': seg_idx,
            'frame_idx': frame_idx,
        }


def calc_ssim(img1, img2, window_size=11):
    """SSIM between tensors in [0,1]"""
    C1, C2 = 0.01**2, 0.03**2
    
    # Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=img1.device)
    coords -= window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    
    window = g.outer(g).unsqueeze(0).unsqueeze(0)
    window = window.expand(3, 1, window_size, window_size)
    
    pad = window_size // 2
    
    mu1 = F.conv2d(img1, window, padding=pad, groups=3)
    mu2 = F.conv2d(img2, window, padding=pad, groups=3)
    
    mu1_sq, mu2_sq = mu1**2, mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1**2, window, padding=pad, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(img2**2, window, padding=pad, groups=3) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=pad, groups=3) - mu1_mu2
    
    ssim = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim.mean().item()


def calc_psnr(img1, img2):
    """PSNR between tensors"""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return 100.0
    return (10 * torch.log10(1.0 / mse)).item()


def tensor_to_pil(t):
    """Convert [1,3,H,W] tensor in [0,1] to PIL"""
    img = t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def create_comparison(gt_pil, pred_pil, metrics, idx):
    """Side-by-side with metrics"""
    w, h = gt_pil.size
    
    # Canvas: GT | Pred with text below
    canvas = Image.new('RGB', (w*2 + 20, h + 50), (255, 255, 255))
    canvas.paste(gt_pil, (0, 0))
    canvas.paste(pred_pil, (w + 20, 0))
    
    draw = ImageDraw.Draw(canvas)
    
    # Labels
    draw.text((w//2 - 30, h + 5), "Ground Truth", fill=(0, 0, 0))
    draw.text((w + 20 + w//2 - 25, h + 5), "Predicted", fill=(0, 0, 0))
    
    # Metrics
    m_str = f"#{idx}  SSIM:{metrics['ssim']:.3f}  PSNR:{metrics['psnr']:.1f}dB"
    if 'lpips' in metrics:
        m_str += f"  LPIPS:{metrics['lpips']:.3f}"
    if 'clip' in metrics:
        m_str += f"  CLIP:{metrics['clip']:.3f}"
    
    draw.text((10, h + 28), m_str, fill=(80, 80, 80))
    
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Adapter checkpoint path')
    parser.add_argument('--preprocessed-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--task', default='dme', choices=['dme', 'tp', 'inscapes'])
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--skip-video', action='store_true')
    parser.add_argument('--skip-clip', action='store_true', help='Skip CLIP similarity computation')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparisons").mkdir(exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    (output_dir / "ground_truth").mkdir(exist_ok=True)
    
    print("=" * 70)
    print("EEGMAMBA + SD 3.5 EVALUATION")
    print("=" * 70)
    
    # Load dataset
    print("\n📂 Loading dataset...")
    dataset = EEGVideoDataset(Path(args.preprocessed_dir), args.task)
    num_samples = min(args.num_samples, len(dataset))
    print(f"  Total samples: {len(dataset):,}")
    print(f"  Evaluating: {num_samples}")
    
    # Load VAE only (we don't need transformer for direct decode)
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
    print(f"\n🧠 Loading adapter from: {args.checkpoint}")
    sample = dataset[0]
    in_channels = sample['eeg'].shape[0]
    eeg_time_steps = sample['eeg'].shape[1]
    
    # FIXED: Use latent_height and latent_width instead of latent_size
    adapter = EEGMambaAdapter(
        in_channels=in_channels,
        eeg_time_steps=eeg_time_steps,
        latent_channels=16,
        latent_height=96,
        latent_width=96,
        use_glmnet=True,
        num_subjects=22,
    ).to(device)
    
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'adapter' in ckpt:
        adapter.load_state_dict(ckpt['adapter'])
    else:
        adapter.load_state_dict(ckpt)
    adapter.eval()
    print(f"  ✓ Loaded (epoch {ckpt.get('epoch', '?')}, loss {ckpt.get('loss', '?'):.4f})")
    
    # Load LPIPS
    lpips_model = None
    if LPIPS_AVAILABLE:
        print("\n📏 Loading LPIPS...")
        lpips_model = lpips.LPIPS(net='alex').to(device)
    
    # Load CLIP (with safetensors fix for torch security)
    clip_model, clip_processor = None, None
    if CLIP_AVAILABLE and not args.skip_clip:
        print("📏 Loading CLIP...")
        try:
            # FIXED: Use safetensors to avoid torch.load security issue (CVE-2025-32434)
            clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                use_safetensors=True
            ).to(device)
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("  ✓ CLIP loaded")
        except Exception as e:
            print(f"  ⚠ CLIP loading failed: {e}")
            print("  → Continuing without CLIP similarity")
            clip_model, clip_processor = None, None
    
    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATING")
    print("=" * 70)
    
    all_metrics = defaultdict(list)
    per_subject_metrics = defaultdict(lambda: defaultdict(list))  # {subject_id: {metric: [values]}}
    per_subject_frames = defaultdict(list)  # {subject_id: [comparison_frames]}
    results = []
    comparison_frames = []
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Samples"):
            sample = dataset[i]
            
            eeg = sample['eeg'].unsqueeze(0).to(device)
            subject_id_tensor = sample['subject_id'].unsqueeze(0).to(device)
            gt_latent = sample['latent'].unsqueeze(0).to(device)
            
            # Predict
            pred_latent = adapter(eeg, subject_id_tensor)
            
            # Decode both
            pred_frame = vae.decode(pred_latent / vae.config.scaling_factor).sample
            gt_frame = vae.decode(gt_latent / vae.config.scaling_factor).sample
            
            # Normalize to [0,1]
            pred_frame = ((pred_frame + 1) / 2).clamp(0, 1)
            gt_frame = ((gt_frame + 1) / 2).clamp(0, 1)
            
            # Metrics
            metrics = {
                'ssim': calc_ssim(pred_frame, gt_frame),
                'psnr': calc_psnr(pred_frame, gt_frame),
            }
            
            # LPIPS
            if lpips_model:
                lpips_score = lpips_model(pred_frame * 2 - 1, gt_frame * 2 - 1)
                metrics['lpips'] = lpips_score.item()
            
            # CLIP similarity
            pred_pil = tensor_to_pil(pred_frame)
            gt_pil = tensor_to_pil(gt_frame)
            
            if clip_model:
                inputs = clip_processor(images=[pred_pil, gt_pil], return_tensors="pt").to(device)
                feats = clip_model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                metrics['clip'] = (feats[0] @ feats[1]).item()
            
            # Store - use the raw subject_id value, not the tensor
            subject_id_val = sample['subject_id'].item()
            for k, v in metrics.items():
                all_metrics[k].append(v)
                per_subject_metrics[subject_id_val][k].append(v)
            
            results.append({
                'idx': i,
                'seg_idx': sample['seg_idx'],
                'frame_idx': sample['frame_idx'],
                'subject_id': subject_id_val,
                **metrics
            })
            
            # Save images
            pred_pil.save(output_dir / "predictions" / f"{i:05d}.png")
            gt_pil.save(output_dir / "ground_truth" / f"{i:05d}.png")
            
            # Comparison
            comp = create_comparison(gt_pil, pred_pil, metrics, i)
            comp.save(output_dir / "comparisons" / f"{i:05d}.png")
            comp_np = np.array(comp)
            comparison_frames.append(comp_np)
            per_subject_frames[subject_id_val].append(comp_np)
    
    # Create video
    if not args.skip_video and comparison_frames:
        print("\n🎬 Creating comparison videos...")
        
        # Overall video
        h, w = comparison_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # AVI format
        out = cv2.VideoWriter(str(output_dir / "comparisons_all.avi"), fourcc, 6, (w, h))
        for frame in comparison_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"  ✓ Overall: {output_dir}/comparisons_all.avi")
        
        # Per-subject videos
        (output_dir / "per_subject_videos").mkdir(exist_ok=True)
        for subj_id in sorted(per_subject_frames.keys()):
            frames = per_subject_frames[subj_id]
            if len(frames) > 0:
                video_path = output_dir / "per_subject_videos" / f"subject_{subj_id:02d}.avi"
                out = cv2.VideoWriter(str(video_path), fourcc, 6, (w, h))
                for frame in frames:
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                out.release()
        print(f"  ✓ Per-subject videos: {output_dir}/per_subject_videos/")
    
    # Summary
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    
    summary = {}
    for k, v in all_metrics.items():
        summary[k] = {
            'mean': np.mean(v),
            'std': np.std(v),
            'min': np.min(v),
            'max': np.max(v),
        }
        arrow = "↑" if k in ['ssim', 'psnr', 'clip'] else "↓"
        print(f"  {k.upper():8s}: {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f} {arrow}")
    
    # Per-subject summary
    print("\n" + "=" * 70)
    print("PER-SUBJECT RESULTS")
    print("=" * 70)
    
    subject_summary = {}
    for subj_id in sorted(per_subject_metrics.keys()):
        subject_summary[int(subj_id)] = {}
        print(f"\n📊 Subject {subj_id}:")
        for k, v in per_subject_metrics[subj_id].items():
            subject_summary[int(subj_id)][k] = {
                'mean': float(np.mean(v)),
                'std': float(np.std(v)),
                'count': len(v),
            }
            arrow = "↑" if k in ['ssim', 'psnr', 'clip'] else "↓"
            print(f"  {k.upper():8s}: {subject_summary[int(subj_id)][k]['mean']:.4f} ± "
                  f"{subject_summary[int(subj_id)][k]['std']:.4f} (n={len(v)}) {arrow}")
    
    # Save JSON
    with open(output_dir / "results.json", 'w') as f:
        json.dump({
            'overall': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in summary.items()},
            'per_subject': subject_summary,
            'per_sample': results,
            'config': vars(args),
        }, f, indent=2)
    
    # Save per-subject CSV summary
    with open(output_dir / "per_subject_summary.csv", 'w', newline='') as f:
        # Get all metric names
        metric_names = sorted(list(summary.keys()))
        fieldnames = ['subject_id', 'num_samples'] + [f"{m}_mean" for m in metric_names] + [f"{m}_std" for m in metric_names]
        
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        
        for subj_id in sorted(subject_summary.keys()):
            row = {'subject_id': subj_id, 'num_samples': subject_summary[subj_id][metric_names[0]]['count']}
            for m in metric_names:
                row[f"{m}_mean"] = f"{subject_summary[subj_id][m]['mean']:.4f}"
                row[f"{m}_std"] = f"{subject_summary[subj_id][m]['std']:.4f}"
            w.writerow(row)
    
    # Save CSV
    with open(output_dir / "metrics.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    
    print(f"\n💾 Results saved to: {output_dir}")
    print("  - results.json (full results with per-subject breakdown)")
    print("  - metrics.csv (per-sample with subject IDs)")
    print("  - per_subject_summary.csv (aggregate stats per subject)")
    print("  - comparisons/ (side-by-side images)")
    print("  - comparisons_all.avi (all samples video)")
    print("  - per_subject_videos/ (one AVI per subject)")
    print("  - predictions/ (generated frames)")
    print("  - ground_truth/ (GT frames)")
    
    print("\n✅ EVALUATION COMPLETE!")


if __name__ == "__main__":
    main()
