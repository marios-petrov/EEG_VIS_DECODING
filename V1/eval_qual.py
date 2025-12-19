#!/usr/bin/env python3
"""
Qualitative evaluation for EEG2Video
Generates videos from EEG signals and creates side-by-side comparisons
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from models import (
    VideoLatentAdapter,
    load_diffusion_models,
    get_inference_scheduler,
)
from utils import (
    EEGVideoDataset,
    tensor_to_pil,
    frames_to_video,
    create_side_by_side_video,
    seed_everything,
    load_checkpoint,
)


class VideoGenerator:
    """Generate videos from EEG using trained adapter"""
    
    def __init__(
        self,
        adapter: torch.nn.Module,
        vae: torch.nn.Module,
        unet: torch.nn.Module,
        scheduler,
        device: torch.device,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
    ):
        self.adapter = adapter
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler
        self.device = device
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        
        self.adapter.eval()
        self.vae.eval()
        self.unet.eval()
    
    @torch.no_grad()
    def generate_from_eeg(
        self,
        eeg: torch.Tensor,
        subject_id: torch.Tensor,
        num_frames: int = 6,
        unconditional_embedding: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate video frames from EEG
        
        Args:
            eeg: [1, C, T] EEG signal
            subject_id: [1] subject ID
            num_frames: Number of frames to generate
            unconditional_embedding: For classifier-free guidance
            
        Returns:
            frames: [num_frames, 3, H, W] generated frames in [-1, 1]
        """
        # Get initial latent from adapter
        initial_latent = self.adapter(eeg, subject_id)  # [1, 4, 96, 96]
        
        # Generate frames with diffusion
        generated_frames = []
        
        for _ in range(num_frames):
            # Start from adapter prediction with added noise
            latent = initial_latent + torch.randn_like(initial_latent) * 0.1
            
            # Prepare empty conditioning (no text prompt)
            encoder_hidden_states = torch.zeros(
                1, 77, 1024, device=self.device
            )
            
            # Denoising loop
            self.scheduler.set_timesteps(self.num_inference_steps)
            
            for t in self.scheduler.timesteps:
                # Predict noise with UNet
                noise_pred = self.unet(
                    latent,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
                
                # Compute previous noisy sample
                latent = self.scheduler.step(
                    noise_pred,
                    t,
                    latent,
                ).prev_sample
            
            # Decode to pixel space
            frame = self.vae.decode(latent / self.vae.config.scaling_factor).sample
            generated_frames.append(frame.squeeze(0))
        
        return torch.stack(generated_frames, dim=0)
    
    @torch.no_grad()
    def generate_video(
        self,
        eeg: torch.Tensor,
        subject_id: torch.Tensor,
        num_frames: int = 6,
    ) -> list:
        """
        Generate video and return as list of PIL Images
        
        Args:
            eeg: [1, C, T] EEG signal
            subject_id: [1] subject ID
            num_frames: Number of frames
            
        Returns:
            List of PIL Images
        """
        frames_tensor = self.generate_from_eeg(eeg, subject_id, num_frames)
        
        # Convert to PIL Images
        frames_pil = [tensor_to_pil(frame) for frame in frames_tensor]
        
        return frames_pil


def evaluate_qualitative(
    adapter_path: Path,
    preprocessed_dir: Path,
    output_dir: Path,
    task: str,
    num_samples: int = 10,
    custom_unet_path: Path = None,
    device: torch.device = torch.device('cuda'),
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    create_comparisons: bool = True,
    save_individual: bool = True,
):
    """
    Run qualitative evaluation
    
    Args:
        adapter_path: Path to trained adapter checkpoint
        preprocessed_dir: Path to preprocessed data
        output_dir: Output directory for generated videos
        task: Task name
        num_samples: Number of samples to generate
        custom_unet_path: Path to fine-tuned UNet (optional)
        device: Device to use
        guidance_scale: CFG scale
        num_inference_steps: Number of denoising steps
        create_comparisons: Create side-by-side comparison videos
        save_individual: Save individual generated videos
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("QUALITATIVE EVALUATION")
    print("=" * 70)
    
    # Load test dataset
    print("\n📊 Loading test dataset...")
    test_dataset = EEGVideoDataset(
        preprocessed_dir=preprocessed_dir,
        task=task,
        split='test',
    )
    print(f"  ✓ Test samples: {len(test_dataset)}")
    
    # Limit samples
    num_samples = min(num_samples, len(test_dataset))
    print(f"  → Generating {num_samples} samples")
    
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
    
    # Get inference scheduler
    scheduler = get_inference_scheduler(
        scheduler_type='ddim',
        num_inference_steps=num_inference_steps,
    )
    
    # Load adapter
    print(f"\n🧠 Loading adapter from: {adapter_path}")
    
    # Infer adapter config from test data
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
    
    # Create generator
    generator = VideoGenerator(
        adapter=adapter,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        device=device,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )
    
    # Generate videos
    print("\n" + "=" * 70)
    print("GENERATING VIDEOS")
    print("=" * 70)
    
    results = []
    
    for i in tqdm(range(num_samples), desc="Generating"):
        sample = test_dataset[i]
        
        # Prepare inputs
        eeg = sample['eeg'].unsqueeze(0).to(device)
        subject_id = sample['subject_id'].unsqueeze(0).to(device)
        gt_latent = sample['latent']  # [frames, 4, 96, 96]
        
        # Generate video
        pred_frames = generator.generate_video(
            eeg=eeg,
            subject_id=subject_id,
            num_frames=gt_latent.shape[0],
        )
        
        # Get ground truth frames
        gt_frames = []
        for frame_latent in gt_latent:
            with torch.no_grad():
                frame_latent = frame_latent.unsqueeze(0).to(device)
                frame = vae.decode(frame_latent / vae.config.scaling_factor).sample
                gt_frames.append(tensor_to_pil(frame.squeeze(0)))
        
        # Convert PIL to numpy for video saving
        pred_frames_np = [np.array(f) for f in pred_frames]
        gt_frames_np = [np.array(f) for f in gt_frames]
        
        # Save individual videos
        if save_individual:
            pred_video_path = output_dir / f"sample_{i:04d}_pred.mp4"
            gt_video_path = output_dir / f"sample_{i:04d}_gt.mp4"
            
            frames_to_video(pred_frames_np, pred_video_path, fps=6)
            frames_to_video(gt_frames_np, gt_video_path, fps=6)
        
        # Create comparison video
        if create_comparisons:
            comparison_path = output_dir / f"sample_{i:04d}_comparison.mp4"
            create_side_by_side_video(
                gt_frames_np,
                pred_frames_np,
                comparison_path,
                fps=6,
                labels=("Ground Truth", "Generated"),
            )
        
        results.append({
            'sample_id': i,
            'index': sample['index'].item(),
        })
    
    # Save results metadata
    results_file = output_dir / "generation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✅ QUALITATIVE EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"Generated {num_samples} videos")
    print(f"Output directory: {output_dir}")
    
    if save_individual:
        print(f"  - Individual videos: sample_XXXX_pred.mp4, sample_XXXX_gt.mp4")
    if create_comparisons:
        print(f"  - Comparison videos: sample_XXXX_comparison.mp4")
    print(f"  - Results metadata: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Qualitative evaluation for EEG2Video")
    
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
                       help='Output directory for generated videos')
    
    # Generation arguments
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of test samples to generate')
    parser.add_argument('--guidance-scale', type=float, default=7.5)
    parser.add_argument('--num-inference-steps', type=int, default=50)
    
    # Output options
    parser.add_argument('--no-comparisons', action='store_true',
                       help='Skip creating comparison videos')
    parser.add_argument('--no-individual', action='store_true',
                       help='Skip saving individual videos')
    
    # System arguments
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Run evaluation
    evaluate_qualitative(
        adapter_path=Path(args.adapter_checkpoint),
        preprocessed_dir=Path(args.preprocessed_dir),
        output_dir=Path(args.output_dir),
        task=args.task,
        num_samples=args.num_samples,
        custom_unet_path=Path(args.unet_checkpoint) if args.unet_checkpoint else None,
        device=device,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        create_comparisons=not args.no_comparisons,
        save_individual=not args.no_individual,
    )


if __name__ == "__main__":
    main()
