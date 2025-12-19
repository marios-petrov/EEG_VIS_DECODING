#!/usr/bin/env python3
"""
Quantitative evaluation for EEG2Video
Calculates metrics: SSIM, PSNR, LPIPS, optical flow consistency
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from models import (
    VideoLatentAdapter,
    load_diffusion_models,
    get_inference_scheduler,
)
from utils import (
    EEGVideoDataset,
    tensor_to_pil,
    calculate_ssim_batch,
    calculate_psnr_batch,
    calculate_lpips_batch,
    calculate_flow_metrics,
    seed_everything,
    load_checkpoint,
    print_metrics,
)


class QuantitativeEvaluator:
    """Evaluate reconstruction quality with quantitative metrics"""
    
    def __init__(
        self,
        adapter: torch.nn.Module,
        vae: torch.nn.Module,
        unet: torch.nn.Module,
        scheduler,
        device: torch.device,
        num_inference_steps: int = 50,
    ):
        self.adapter = adapter
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler
        self.device = device
        self.num_inference_steps = num_inference_steps
        
        # Load LPIPS model
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            self.lpips_available = True
        except Exception as e:
            print(f"⚠️  LPIPS not available: {e}")
            self.lpips_model = None
            self.lpips_available = False
        
        self.adapter.eval()
        self.vae.eval()
        self.unet.eval()
    
    @torch.no_grad()
    def generate_frames(
        self,
        eeg: torch.Tensor,
        subject_id: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """
        Generate video frames from EEG
        
        Returns:
            frames: [num_frames, 3, H, W] in [-1, 1]
        """
        # Get initial latent from adapter
        initial_latent = self.adapter(eeg, subject_id)
        
        # Generate frames
        generated_frames = []
        
        for _ in range(num_frames):
            latent = initial_latent + torch.randn_like(initial_latent) * 0.1
            
            # Empty conditioning
            encoder_hidden_states = torch.zeros(1, 77, 1024, device=self.device)
            
            # Denoising
            self.scheduler.set_timesteps(self.num_inference_steps)
            for t in self.scheduler.timesteps:
                noise_pred = self.unet(
                    latent, t, encoder_hidden_states=encoder_hidden_states
                ).sample
                latent = self.scheduler.step(noise_pred, t, latent).prev_sample
            
            # Decode
            frame = self.vae.decode(latent / self.vae.config.scaling_factor).sample
            generated_frames.append(frame.squeeze(0))
        
        return torch.stack(generated_frames, dim=0)
    
    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representations to pixel space
        
        Args:
            latents: [num_frames, 4, 96, 96]
        Returns:
            frames: [num_frames, 3, H, W] in [-1, 1]
        """
        frames = []
        for latent in latents:
            latent = latent.unsqueeze(0).to(self.device)
            frame = self.vae.decode(latent / self.vae.config.scaling_factor).sample
            frames.append(frame.squeeze(0))
        return torch.stack(frames, dim=0)
    
    def compute_metrics(
        self,
        pred_frames: torch.Tensor,
        gt_frames: torch.Tensor,
    ) -> dict:
        """
        Compute all metrics between predicted and ground truth frames
        
        Args:
            pred_frames: [num_frames, 3, H, W] in [-1, 1]
            gt_frames: [num_frames, 3, H, W] in [-1, 1]
            
        Returns:
            Dict of metrics
        """
        metrics = {}
        
        # Normalize to [0, 1] for SSIM and PSNR
        pred_norm = (pred_frames + 1) / 2
        gt_norm = (gt_frames + 1) / 2
        
        # SSIM (per frame, then average)
        ssim_scores = []
        for pred, gt in zip(pred_norm, gt_norm):
            ssim = calculate_ssim_batch(
                pred.unsqueeze(0),
                gt.unsqueeze(0),
                size_average=True,
            )
            ssim_scores.append(ssim.item())
        metrics['ssim_mean'] = np.mean(ssim_scores)
        metrics['ssim_std'] = np.std(ssim_scores)
        
        # PSNR (per frame, then average)
        psnr_scores = []
        for pred, gt in zip(pred_norm, gt_norm):
            psnr = calculate_psnr_batch(
                pred.unsqueeze(0),
                gt.unsqueeze(0),
                max_val=1.0,
            )
            psnr_scores.append(psnr.item())
        metrics['psnr_mean'] = np.mean(psnr_scores)
        metrics['psnr_std'] = np.std(psnr_scores)
        
        # LPIPS (per frame, then average)
        if self.lpips_available:
            lpips_scores = []
            for pred, gt in zip(pred_frames, gt_frames):
                lpips_score = calculate_lpips_batch(
                    pred.unsqueeze(0),
                    gt.unsqueeze(0),
                    self.lpips_model,
                )
                lpips_scores.append(lpips_score.item())
            metrics['lpips_mean'] = np.mean(lpips_scores)
            metrics['lpips_std'] = np.std(lpips_scores)
        
        # Optical flow metrics
        # Convert to numpy [H, W, 3] in [0, 255]
        pred_np = [
            ((f.cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
            for f in pred_frames
        ]
        gt_np = [
            ((f.cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
            for f in gt_frames
        ]
        
        flow_metrics = calculate_flow_metrics(gt_np, pred_np)
        metrics.update(flow_metrics)
        
        return metrics
    
    def evaluate_sample(
        self,
        sample: dict,
    ) -> dict:
        """Evaluate a single sample"""
        # Prepare inputs
        eeg = sample['eeg'].unsqueeze(0).to(self.device)
        subject_id = sample['subject_id'].unsqueeze(0).to(self.device)
        gt_latents = sample['latent'].to(self.device)
        
        # Generate frames
        pred_frames = self.generate_frames(
            eeg=eeg,
            subject_id=subject_id,
            num_frames=gt_latents.shape[0],
        )
        
        # Decode ground truth
        gt_frames = self.decode_latents(gt_latents)
        
        # Compute metrics
        metrics = self.compute_metrics(pred_frames, gt_frames)
        
        return metrics


def evaluate_quantitative(
    adapter_path: Path,
    preprocessed_dir: Path,
    output_dir: Path,
    task: str,
    num_samples: int = None,
    custom_unet_path: Path = None,
    device: torch.device = torch.device('cuda'),
    num_inference_steps: int = 50,
):
    """
    Run quantitative evaluation on test set
    
    Args:
        adapter_path: Path to trained adapter checkpoint
        preprocessed_dir: Path to preprocessed data
        output_dir: Output directory for results
        task: Task name
        num_samples: Number of samples to evaluate (None = all)
        custom_unet_path: Path to fine-tuned UNet
        device: Device to use
        num_inference_steps: Number of denoising steps
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("QUANTITATIVE EVALUATION")
    print("=" * 70)
    
    # Load test dataset
    print("\n📊 Loading test dataset...")
    test_dataset = EEGVideoDataset(
        preprocessed_dir=preprocessed_dir,
        task=task,
        split='test',
    )
    print(f"  ✓ Test samples: {len(test_dataset)}")
    
    # Limit samples if specified
    if num_samples is not None:
        num_samples = min(num_samples, len(test_dataset))
        print(f"  → Evaluating {num_samples} samples")
    else:
        num_samples = len(test_dataset)
    
    # Load diffusion models
    print("\n🎨 Loading diffusion models...")
    vae, unet, _, _, _ = load_diffusion_models(
        device=device,
        load_vae=True,
        load_unet=True,
        load_text_encoder=False,
        custom_unet_path=str(custom_unet_path) if custom_unet_path else None,
    )
    
    if custom_unet_path:
        print(f"  ✓ Loaded fine-tuned UNet from: {custom_unet_path}")
    
    # Get scheduler
    scheduler = get_inference_scheduler(
        scheduler_type='ddim',
        num_inference_steps=num_inference_steps,
    )
    
    # Load adapter
    print(f"\n🧠 Loading adapter from: {adapter_path}")
    
    sample = test_dataset[0]
    in_channels = sample['eeg'].shape[0]
    eeg_time_steps = sample['eeg'].shape[1]
    
    adapter = VideoLatentAdapter(
        in_channels=in_channels,
        eeg_time_steps=eeg_time_steps,
        latent_height=96,
        latent_width=96,
        latent_channels=4,
    ).to(device)
    
    load_checkpoint(adapter_path, adapter, device=device)
    print(f"  ✓ Adapter loaded")
    
    # Create evaluator
    evaluator = QuantitativeEvaluator(
        adapter=adapter,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        device=device,
        num_inference_steps=num_inference_steps,
    )
    
    # Evaluate samples
    print("\n" + "=" * 70)
    print("COMPUTING METRICS")
    print("=" * 70)
    
    all_metrics = defaultdict(list)
    sample_results = []
    
    for i in tqdm(range(num_samples), desc="Evaluating"):
        sample = test_dataset[i]
        
        # Evaluate sample
        metrics = evaluator.evaluate_sample(sample)
        
        # Store per-sample results
        sample_results.append({
            'sample_id': i,
            'index': sample['index'].item(),
            **metrics,
        })
        
        # Accumulate for averaging
        for key, value in metrics.items():
            if not np.isnan(value):
                all_metrics[key].append(value)
    
    # Compute aggregate statistics
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    aggregate_results = {}
    for key, values in all_metrics.items():
        aggregate_results[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
        }
    
    # Print summary
    print("\n📊 Summary Statistics:")
    print("-" * 60)
    for metric_name in ['ssim_mean', 'psnr_mean', 'lpips_mean', 'flow_corr']:
        if metric_name in aggregate_results:
            stats = aggregate_results[metric_name]
            print(f"{metric_name:20s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"{'':20s}  (min: {stats['min']:.4f}, max: {stats['max']:.4f})")
    print("-" * 60)
    
    # Save results
    results_file = output_dir / "quantitative_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'aggregate': aggregate_results,
            'per_sample': sample_results,
            'num_samples': num_samples,
            'task': task,
        }, f, indent=2)
    
    print(f"\n💾 Saved results to: {results_file}")
    
    # Save summary CSV for easy import
    import csv
    csv_file = output_dir / "quantitative_summary.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'mean', 'std', 'min', 'max', 'median'])
        for metric_name, stats in aggregate_results.items():
            writer.writerow([
                metric_name,
                stats['mean'],
                stats['std'],
                stats['min'],
                stats['max'],
                stats['median'],
            ])
    
    print(f"💾 Saved summary to: {csv_file}")
    
    print("\n" + "=" * 70)
    print("✅ QUANTITATIVE EVALUATION COMPLETE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Quantitative evaluation for EEG2Video")
    
    # Model arguments
    parser.add_argument('--adapter-checkpoint', required=True,
                       help='Path to trained adapter checkpoint')
    parser.add_argument('--unet-checkpoint',
                       help='Path to fine-tuned UNet checkpoint (optional)')
    
    # Data arguments
    parser.add_argument('--preprocessed-dir', required=True,
                       help='Preprocessed data directory')
    parser.add_argument('--task', required=True, choices=['dme', 'tp', 'inscapes'])
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for results')
    
    # Evaluation arguments
    parser.add_argument('--num-samples', type=int,
                       help='Number of test samples to evaluate (default: all)')
    parser.add_argument('--num-inference-steps', type=int, default=50)
    
    # System arguments
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Run evaluation
    evaluate_quantitative(
        adapter_path=Path(args.adapter_checkpoint),
        preprocessed_dir=Path(args.preprocessed_dir),
        output_dir=Path(args.output_dir),
        task=args.task,
        num_samples=args.num_samples,
        custom_unet_path=Path(args.unet_checkpoint) if args.unet_checkpoint else None,
        device=device,
        num_inference_steps=args.num_inference_steps,
    )


if __name__ == "__main__":
    main()
