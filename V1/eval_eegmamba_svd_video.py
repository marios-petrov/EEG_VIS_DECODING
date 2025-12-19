#!/usr/bin/env python3
"""
Evaluate EEGMamba + SVD for video reconstruction on SEED-DV dataset.

Pipeline:
1. EEG → EEGMamba → predicted frame latent
2. Decode latent → blurry first frame  
3. (Optional) Refine with SD3 img2img → sharp first frame
4. First frame → Stable Video Diffusion → video

Usage:
    python eval_eegmamba_svd_video.py \
        --checkpoint /path/to/adapter_best.pt \
        --preprocessed-dir /path/to/preprocessed \
        --output-dir /path/to/eval_results \
        --max-samples 10
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


class SEEDDVVideoDataset(Dataset):
    """SEED-DV test dataset with full video latents"""
    
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
        all_latents = self.latent_data[subj_id][seg_idx]  # All frames
        
        if self.precomputed_embeddings is not None:
            caption = seg['caption']
            caption_emb = self.precomputed_embeddings.get(caption, torch.zeros(2048))
        else:
            caption_emb = torch.zeros(2048)
        
        return {
            'eeg': torch.from_numpy(eeg).float(),
            'all_latents': torch.from_numpy(all_latents).float(),  # [N, C, H, W]
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


def decode_latents_to_video(latents: torch.Tensor, vae, device) -> List[np.ndarray]:
    """Decode multiple latents to video frames"""
    frames = []
    for i in range(len(latents)):
        frame = decode_latent(latents[i], vae, device)
        frames.append(frame)
    return frames


def load_svd_pipeline(device):
    """Load Stable Video Diffusion pipeline"""
    from diffusers import StableVideoDiffusionPipeline
    
    print("Loading Stable Video Diffusion pipeline...")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    
    # Enable memory optimizations
    pipe.enable_model_cpu_offload()
    
    print("  ✓ SVD pipeline loaded")
    return pipe


def generate_video_svd(
    pipe,
    first_frame: np.ndarray,
    num_frames: int = 14,
    fps: int = 7,
    motion_bucket_id: int = 127,
    noise_aug_strength: float = 0.02,
    num_inference_steps: int = 25,
) -> List[np.ndarray]:
    """Generate video from first frame using SVD"""
    
    # SVD expects 1024x576 or 576x1024 images
    # Resize input frame
    pil_image = Image.fromarray(first_frame).convert("RGB")
    pil_image = pil_image.resize((1024, 576), Image.LANCZOS)
    
    with torch.no_grad():
        frames = pipe(
            image=pil_image,
            num_frames=num_frames,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            num_inference_steps=num_inference_steps,
            decode_chunk_size=4,
        ).frames[0]
    
    # Convert to numpy arrays
    video_frames = [np.array(f) for f in frames]
    
    return video_frames


def save_video(frames: List[np.ndarray], output_path: Path, fps: int = 7):
    """Save frames as MP4 video"""
    if len(frames) == 0:
        return
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    
    writer.release()


def save_frames_grid(frames: List[np.ndarray], output_path: Path, cols: int = 7):
    """Save frames as a grid image"""
    if len(frames) == 0:
        return
    
    n_frames = len(frames)
    rows = (n_frames + cols - 1) // cols
    
    h, w = frames[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    
    for i, frame in enumerate(frames):
        r, c = i // cols, i % cols
        # Resize frame to match first frame if needed
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = frame
    
    Image.fromarray(grid).save(output_path)


def compute_video_ssim(pred_frames: List[np.ndarray], gt_frames: List[np.ndarray]) -> float:
    """Compute average SSIM across video frames"""
    # Match frame counts by sampling
    n_pred = len(pred_frames)
    n_gt = len(gt_frames)
    
    ssim_values = []
    
    # Sample frames at matching temporal positions
    n_compare = min(n_pred, n_gt)
    for i in range(n_compare):
        pred_idx = int(i * n_pred / n_compare)
        gt_idx = int(i * n_gt / n_compare)
        
        pred_frame = pred_frames[pred_idx]
        gt_frame = gt_frames[gt_idx]
        
        # Resize if needed
        if pred_frame.shape[:2] != gt_frame.shape[:2]:
            pred_frame = cv2.resize(pred_frame, (gt_frame.shape[1], gt_frame.shape[0]))
        
        # Convert to grayscale
        pred_gray = cv2.cvtColor(pred_frame, cv2.COLOR_RGB2GRAY)
        gt_gray = cv2.cvtColor(gt_frame, cv2.COLOR_RGB2GRAY)
        
        ssim_val = ssim(pred_gray, gt_gray, data_range=255)
        ssim_values.append(ssim_val)
    
    return float(np.mean(ssim_values))


def load_clip_classifier(device):
    from transformers import CLIPProcessor, CLIPModel
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        torch_dtype=torch.float16,
    ).to(device).eval()
    return model, processor


def classify_video_clip(
    frames: List[np.ndarray],
    model,
    processor,
    device,
    concepts: List[str] = SEED_DV_CONCEPTS,
) -> Tuple[int, np.ndarray]:
    """Classify video by averaging CLIP predictions across frames"""
    
    text_prompts = [f"a video of {concept}" for concept in concepts]
    
    all_probs = []
    
    # Sample up to 5 frames for classification
    n_frames = len(frames)
    sample_indices = [int(i * n_frames / 5) for i in range(min(5, n_frames))]
    
    for idx in sample_indices:
        pil_image = Image.fromarray(frames[idx])
        
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
        
        all_probs.append(probs)
    
    # Average predictions
    avg_probs = np.mean(all_probs, axis=0)
    
    return int(np.argmax(avg_probs)), avg_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--preprocessed-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--subjects', type=str, default='1-20')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no-captions', action='store_true')
    # SVD settings
    parser.add_argument('--num-frames', type=int, default=14, help='Number of video frames to generate')
    parser.add_argument('--svd-steps', type=int, default=25, help='SVD inference steps')
    parser.add_argument('--motion-bucket', type=int, default=127, help='Motion bucket ID (0-255, higher=more motion)')
    # Optional SD3 refinement before SVD
    parser.add_argument('--refine-first-frame', action='store_true', 
                       help='Refine first frame with SD3 img2img before SVD')
    parser.add_argument('--refine-strength', type=float, default=0.4)
    parser.add_argument('--refine-steps', type=int, default=15)
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
    svd_pipe = load_svd_pipeline(device)
    
    # Optional: Load SD3 for first frame refinement
    sd3_pipe = None
    if args.refine_first_frame:
        from diffusers import StableDiffusion3Img2ImgPipeline
        print("Loading SD3 img2img for first frame refinement...")
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
    test_dataset = SEEDDVVideoDataset(
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
    
    # Setup output directory
    use_captions = not args.no_captions
    caption_str = "with_captions" if use_captions else "no_captions"
    output_dir = Path(args.output_dir) / f"video_{caption_str}_svd{args.num_frames}frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "videos_pred").mkdir(exist_ok=True)
    (output_dir / "videos_gt").mkdir(exist_ok=True)
    (output_dir / "grids_pred").mkdir(exist_ok=True)
    (output_dir / "grids_gt").mkdir(exist_ok=True)
    (output_dir / "first_frames").mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Video Generation Settings:")
    print(f"  SVD frames: {args.num_frames}")
    print(f"  SVD steps: {args.svd_steps}")
    print(f"  Motion bucket: {args.motion_bucket}")
    print(f"  Refine first frame: {args.refine_first_frame}")
    print(f"  Captions: {'enabled' if use_captions else 'DISABLED'}")
    print(f"{'='*60}\n")
    
    # Run evaluation
    all_results = []
    correct_40way = 0
    correct_2way = 0
    total_samples = 0
    
    max_samples = args.max_samples or len(test_dataset)
    
    for batch_idx, batch in enumerate(tqdm(test_loader, total=min(max_samples, len(test_loader)))):
        if batch_idx >= max_samples:
            break
        
        eeg = batch['eeg'].to(device)
        all_latents = batch['all_latents'][0]  # [N, C, H, W]
        caption_emb = batch['caption_emb'].to(device)
        concept_id = batch['concept_id'].item()
        subject_id = batch['subject_id'].item()
        caption = batch['caption'][0]
        
        if not use_captions:
            caption_emb = torch.zeros_like(caption_emb)
        
        # Generate predicted first frame from EEG
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred_latent = model(eeg, caption_emb)
        
        # Decode to first frame (blurry)
        first_frame = decode_latent(pred_latent[0], vae, device)
        
        # Optional: Refine first frame with SD3
        if args.refine_first_frame and sd3_pipe is not None:
            prompt = caption if use_captions else ""
            pil_image = Image.fromarray(first_frame).convert("RGB")
            with torch.no_grad():
                refined = sd3_pipe(
                    prompt=prompt,
                    image=pil_image,
                    strength=args.refine_strength,
                    num_inference_steps=args.refine_steps,
                    guidance_scale=4.5,
                ).images[0]
            first_frame = np.array(refined)
        
        # Generate video from first frame using SVD
        try:
            pred_video = generate_video_svd(
                svd_pipe,
                first_frame,
                num_frames=args.num_frames,
                motion_bucket_id=args.motion_bucket,
                num_inference_steps=args.svd_steps,
            )
        except Exception as e:
            print(f"SVD failed for sample {batch_idx}: {e}")
            pred_video = [first_frame] * args.num_frames
        
        # Decode ground truth video
        gt_video = decode_latents_to_video(all_latents, vae, device)
        
        # Compute video SSIM
        video_ssim = compute_video_ssim(pred_video, gt_video)
        
        # Classify video
        pred_class, pred_probs = classify_video_clip(
            pred_video, clip_model, clip_processor, device
        )
        
        if pred_class == concept_id:
            correct_40way += 1
        
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
            'video_ssim': video_ssim,
            'correct_40way': bool(pred_class == concept_id),
            'correct_2way': bool(gt_is_animal == pred_is_animal),
            'n_pred_frames': len(pred_video),
            'n_gt_frames': len(gt_video),
        }
        all_results.append(result)
        
        # Save videos and grids
        if batch_idx < 50:
            # Save first frame
            Image.fromarray(first_frame).save(
                output_dir / "first_frames" / f"{batch_idx:05d}.png"
            )
            
            # Save predicted video
            save_video(pred_video, output_dir / "videos_pred" / f"{batch_idx:05d}.mp4")
            save_frames_grid(pred_video, output_dir / "grids_pred" / f"{batch_idx:05d}.png")
            
            # Save GT video
            save_video(gt_video, output_dir / "videos_gt" / f"{batch_idx:05d}.mp4")
            save_frames_grid(gt_video, output_dir / "grids_gt" / f"{batch_idx:05d}.png")
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{max_samples}, "
                  f"40-way acc: {correct_40way/(batch_idx+1)*100:.1f}%, "
                  f"Video SSIM: {np.mean([r['video_ssim'] for r in all_results]):.3f}")
    
    # Summary
    ssim_values = [r['video_ssim'] for r in all_results]
    
    summary = {
        'total_samples': total_samples,
        'settings': {
            'num_frames': args.num_frames,
            'svd_steps': args.svd_steps,
            'motion_bucket': args.motion_bucket,
            'refine_first_frame': args.refine_first_frame,
            'use_captions': use_captions,
        },
        'metrics': {
            'video_ssim_mean': float(np.mean(ssim_values)),
            'video_ssim_std': float(np.std(ssim_values)),
            'acc_40way': float(correct_40way / total_samples),
            'acc_2way': float(correct_2way / total_samples),
        },
    }
    
    # Print results
    caption_mode = "WITH captions" if use_captions else "WITHOUT captions"
    print("\n" + "="*60)
    print(f"VIDEO EVALUATION RESULTS ({caption_mode})")
    print("="*60)
    print(f"\nSettings: {args.num_frames} frames, {args.svd_steps} SVD steps")
    print(f"Total samples: {total_samples}")
    print(f"\nVideo-level metrics:")
    print(f"  Video SSIM: {summary['metrics']['video_ssim_mean']:.3f} ± {summary['metrics']['video_ssim_std']:.3f}")
    print(f"\nSemantic-level metrics:")
    print(f"  40-way accuracy: {summary['metrics']['acc_40way']*100:.1f}%")
    print(f"  2-way accuracy: {summary['metrics']['acc_2way']*100:.1f}%")
    
    print("\n" + "="*60)
    print("COMPARISON WITH EEG2Video (NeurIPS 2024)")
    print("="*60)
    print(f"{'Metric':<20} {'EEG2Video':<15} {'Ours (SVD)':<20}")
    print("-"*55)
    print(f"{'SSIM':<20} {'0.256 ± 0.03':<15} {summary['metrics']['video_ssim_mean']:.3f} ± {summary['metrics']['video_ssim_std']:.2f}")
    print(f"{'40-way acc':<20} {'15.9%':<15} {summary['metrics']['acc_40way']*100:.1f}%")
    print(f"{'2-way acc':<20} {'79.8%':<15} {summary['metrics']['acc_2way']*100:.1f}%")
    
    # Save results
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - videos_pred/: Generated videos")
    print(f"  - videos_gt/: Ground truth videos")
    print(f"  - grids_pred/: Predicted frame grids")
    print(f"  - grids_gt/: GT frame grids")


if __name__ == "__main__":
    main()
