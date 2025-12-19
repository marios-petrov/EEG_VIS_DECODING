#!/usr/bin/env python3
"""
SEED-DV Image Quality Evaluation (SSIM, PSNR, LPIPS)

Evaluates pixel-level reconstruction quality:
- SSIM: Structural Similarity Index (paper reports 0.256)
- PSNR: Peak Signal-to-Noise Ratio
- LPIPS: Learned Perceptual Image Patch Similarity

Usage:
    python eval_seed_dv_ssim.py \
        --generated-dir /path/to/stage1_results \
        --gt-dir /path/to/preprocessed_data \
        --output-dir /path/to/eval_output
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
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Try to import metrics
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: skimage not available, will use torch implementation")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available, skipping LPIPS metric")


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two images"""
    if SKIMAGE_AVAILABLE:
        # Convert to grayscale for SSIM (standard approach)
        if img1.ndim == 3:
            img1_gray = np.mean(img1, axis=2)
            img2_gray = np.mean(img2, axis=2)
        else:
            img1_gray = img1
            img2_gray = img2
        
        return ssim(img1_gray, img2_gray, data_range=255)
    else:
        # Simple torch-based SSIM approximation
        return compute_ssim_torch(img1, img2)


def compute_ssim_torch(img1: np.ndarray, img2: np.ndarray, window_size: int = 11) -> float:
    """Torch-based SSIM computation"""
    img1 = torch.from_numpy(img1).float().unsqueeze(0).unsqueeze(0)
    img2 = torch.from_numpy(img2).float().unsqueeze(0).unsqueeze(0)
    
    if img1.dim() == 5:  # RGB
        img1 = img1.mean(dim=2)
        img2 = img2.mean(dim=2)
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    mu1 = torch.nn.functional.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = torch.nn.functional.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = torch.nn.functional.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = torch.nn.functional.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = torch.nn.functional.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images"""
    if SKIMAGE_AVAILABLE:
        return psnr(img1, img2, data_range=255)
    else:
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))


class LPIPSMetric:
    """LPIPS metric wrapper"""
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        
    def load(self):
        if LPIPS_AVAILABLE and self.model is None:
            self.model = lpips.LPIPS(net='alex').to(self.device)
            self.model.eval()
    
    def compute(self, img1: np.ndarray, img2: np.ndarray) -> Optional[float]:
        if not LPIPS_AVAILABLE or self.model is None:
            return None
        
        # Convert to tensor [1, 3, H, W] in range [-1, 1]
        def to_tensor(img):
            t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
            t = t / 127.5 - 1.0
            return t.to(self.device)
        
        with torch.no_grad():
            dist = self.model(to_tensor(img1), to_tensor(img2))
        
        return dist.item()


def load_ground_truth_frames(
    preprocessed_dir: Path,
    subject_ids: List[int],
    test_only: bool = True,
) -> Dict[str, Tuple[np.ndarray, dict]]:
    """Load ground truth frames from preprocessed data"""
    gt_data = {}
    
    for subj_id in subject_ids:
        subj_dir = preprocessed_dir / f"sub{subj_id}"
        if not subj_dir.exists():
            continue
        
        meta_path = subj_dir / "metadata.json"
        if not meta_path.exists():
            continue
        
        with open(meta_path) as f:
            metadata = json.load(f)
        
        # Load frames if available
        frames_path = subj_dir / "frames.npy"
        latents_path = subj_dir / "latents.npy"
        
        # We might need to decode latents to frames
        segments = metadata.get('segments', [])
        
        for seg in segments:
            if test_only and seg.get('is_train', True):
                continue
            
            key = f"{subj_id}_{seg['segment_idx']}"
            gt_data[key] = {
                'subject_id': subj_id,
                'segment_idx': seg['segment_idx'],
                'concept_id': seg['concept_id'],
                'concept_name': seg.get('concept_name', 'unknown'),
                'is_train': seg.get('is_train', True),
            }
    
    return gt_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated-dir', type=str, required=True,
                       help='Directory with generated images (stage1 results)')
    parser.add_argument('--gt-dir', type=str, default=None,
                       help='Directory with ground truth frames (optional)')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use-blurry', action='store_true',
                       help='Evaluate blurry frames instead of refined')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--compute-lpips', action='store_true',
                       help='Compute LPIPS (slower)')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    gen_dir = Path(args.generated_dir)
    
    # Load metadata
    meta_path = gen_dir / "metadata.json"
    if not meta_path.exists():
        print(f"ERROR: metadata.json not found in {gen_dir}")
        return
    
    with open(meta_path) as f:
        metadata = json.load(f)
    
    samples = metadata['samples']
    settings = metadata.get('settings', {})
    
    print(f"\n{'='*60}")
    print("SEED-DV IMAGE QUALITY EVALUATION")
    print(f"{'='*60}")
    print(f"Generated dir: {gen_dir}")
    print(f"Samples: {len(samples)}")
    print(f"Frame type: {'blurry' if args.use_blurry else 'refined'}")
    
    # Output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = "_blurry" if args.use_blurry else "_refined"
        output_dir = gen_dir.parent / f"eval_ssim{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize LPIPS if requested
    lpips_metric = None
    if args.compute_lpips:
        lpips_metric = LPIPSMetric(device)
        lpips_metric.load()
    
    # Check if we have GT frames
    gt_frames_dir = gen_dir / "gt_frames"
    has_gt = gt_frames_dir.exists()
    
    if not has_gt:
        print("\nNOTE: No GT frames found. Computing SSIM between generated and GT requires")
        print("      GT frames in generated-dir/gt_frames/ or --gt-dir with frames.npy")
    
    # Evaluate
    results = []
    ssim_scores = []
    psnr_scores = []
    lpips_scores = []
    
    concept_stats = defaultdict(lambda: {"ssim": [], "psnr": [], "lpips": []})
    
    max_samples = args.max_samples or len(samples)
    
    print(f"\nEvaluating {min(max_samples, len(samples))} samples...")
    
    for sample in tqdm(samples[:max_samples]):
        idx = sample['sample_idx']
        concept_id = sample['concept_id']
        
        # Load generated frame
        if args.use_blurry:
            gen_path = gen_dir / "blurry" / f"{idx:05d}.png"
        else:
            gen_path = gen_dir / "refined" / f"{idx:05d}.png"
        
        if not gen_path.exists():
            continue
        
        gen_img = np.array(Image.open(gen_path))
        
        # Load GT frame if available
        gt_path = gt_frames_dir / f"{idx:05d}.png"
        if not gt_path.exists():
            gt_path = gen_dir / "gt" / f"{idx:05d}.png"
        
        if gt_path.exists():
            gt_img = np.array(Image.open(gt_path))
            
            # Resize if needed
            if gen_img.shape != gt_img.shape:
                from PIL import Image as PILImage
                gt_img = np.array(PILImage.fromarray(gt_img).resize(
                    (gen_img.shape[1], gen_img.shape[0]), PILImage.BILINEAR
                ))
            
            # Compute metrics
            ssim_val = compute_ssim(gen_img, gt_img)
            psnr_val = compute_psnr(gen_img, gt_img)
            
            ssim_scores.append(ssim_val)
            psnr_scores.append(psnr_val)
            
            concept_stats[concept_id]["ssim"].append(ssim_val)
            concept_stats[concept_id]["psnr"].append(psnr_val)
            
            # LPIPS
            lpips_val = None
            if lpips_metric is not None:
                lpips_val = lpips_metric.compute(gen_img, gt_img)
                if lpips_val is not None:
                    lpips_scores.append(lpips_val)
                    concept_stats[concept_id]["lpips"].append(lpips_val)
            
            result = {
                'sample_idx': idx,
                'concept_id': concept_id,
                'ssim': float(ssim_val),
                'psnr': float(psnr_val),
                'lpips': float(lpips_val) if lpips_val else None,
            }
        else:
            result = {
                'sample_idx': idx,
                'concept_id': concept_id,
                'ssim': None,
                'psnr': None,
                'lpips': None,
                'note': 'GT not available',
            }
        
        results.append(result)
    
    # Compute statistics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    if ssim_scores:
        mean_ssim = np.mean(ssim_scores)
        std_ssim = np.std(ssim_scores)
        print(f"\nSSIM:  {mean_ssim:.4f} ± {std_ssim:.4f}")
        print(f"       (Paper reports: 0.256)")
    else:
        mean_ssim = None
        std_ssim = None
        print("\nSSIM: N/A (no GT frames)")
    
    if psnr_scores:
        mean_psnr = np.mean(psnr_scores)
        std_psnr = np.std(psnr_scores)
        print(f"\nPSNR:  {mean_psnr:.2f} ± {std_psnr:.2f} dB")
    else:
        mean_psnr = None
        std_psnr = None
    
    if lpips_scores:
        mean_lpips = np.mean(lpips_scores)
        std_lpips = np.std(lpips_scores)
        print(f"\nLPIPS: {mean_lpips:.4f} ± {std_lpips:.4f}")
        print(f"       (lower is better)")
    else:
        mean_lpips = None
        std_lpips = None
    
    # Per-concept breakdown
    if concept_stats:
        print("\n" + "="*60)
        print("PER-CONCEPT SSIM (Top 5 Best)")
        print("="*60)
        
        concept_ssim = []
        for cid, stats in concept_stats.items():
            if stats["ssim"]:
                avg = np.mean(stats["ssim"])
                concept_ssim.append((cid, avg, len(stats["ssim"])))
        
        concept_ssim.sort(key=lambda x: x[1], reverse=True)
        
        print(f"{'Concept ID':<12} {'SSIM':<10} {'N'}")
        print("-"*30)
        for cid, avg, n in concept_ssim[:5]:
            print(f"{cid:<12} {avg:.4f}{'':<4} {n}")
        
        print("\nBottom 5:")
        for cid, avg, n in concept_ssim[-5:]:
            print(f"{cid:<12} {avg:.4f}{'':<4} {n}")
    
    # Save results
    summary = {
        'metrics': {
            'ssim_mean': float(mean_ssim) if mean_ssim else None,
            'ssim_std': float(std_ssim) if std_ssim else None,
            'psnr_mean': float(mean_psnr) if mean_psnr else None,
            'psnr_std': float(std_psnr) if std_psnr else None,
            'lpips_mean': float(mean_lpips) if mean_lpips else None,
            'lpips_std': float(std_lpips) if std_lpips else None,
        },
        'paper_reference': {
            'ssim': 0.256,
            'note': 'EEG2Video paper Table 2',
        },
        'settings': {
            'generated_dir': str(gen_dir),
            'use_blurry': args.use_blurry,
            'has_gt': has_gt,
            'n_evaluated': len(ssim_scores) if ssim_scores else 0,
        },
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
