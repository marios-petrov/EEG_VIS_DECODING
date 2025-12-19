#!/usr/bin/env python3
"""
Stage 2: Generate videos from refined frames using SVD.

Run after stage 1 (eval_stage1_refine.py) which saves refined frames to disk.

Usage:
    python eval_stage2_svd.py \
        --stage1-dir /path/to/stage1_with_captions_str0.5 \
        --output-dir /path/to/eval_results
"""

import os
os.environ['HF_HOME'] = '/local-scratch/marios-datasets/hf-cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/local-scratch/marios-datasets/hf-cache/hub'

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import cv2
from typing import List, Tuple
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


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
    
    # SVD expects 1024x576 images
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
    
    return [np.array(f) for f in frames]


def save_video(frames: List[np.ndarray], output_path: Path, fps: int = 7):
    """Save frames as MP4"""
    if len(frames) == 0:
        return
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    
    writer.release()


def save_frames_grid(frames: List[np.ndarray], output_path: Path, cols: int = 7):
    """Save frames as grid image"""
    n_frames = len(frames)
    rows = (n_frames + cols - 1) // cols
    
    h, w = frames[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    
    for i, frame in enumerate(frames):
        r, c = i // cols, i % cols
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = frame
    
    Image.fromarray(grid).save(output_path)


def load_gt_frames(gt_dir: Path) -> List[np.ndarray]:
    """Load GT frames from directory"""
    frames = []
    frame_files = sorted(gt_dir.glob("frame_*.png"))
    for f in frame_files:
        frames.append(np.array(Image.open(f)))
    return frames


def compute_video_ssim(pred_frames: List[np.ndarray], gt_frames: List[np.ndarray]) -> float:
    """Compute average SSIM across matched frames"""
    n_pred = len(pred_frames)
    n_gt = len(gt_frames)
    n_compare = min(n_pred, n_gt)
    
    ssim_values = []
    for i in range(n_compare):
        pred_idx = int(i * n_pred / n_compare)
        gt_idx = int(i * n_gt / n_compare)
        
        pred_frame = pred_frames[pred_idx]
        gt_frame = gt_frames[gt_idx]
        
        if pred_frame.shape[:2] != gt_frame.shape[:2]:
            pred_frame = cv2.resize(pred_frame, (gt_frame.shape[1], gt_frame.shape[0]))
        
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
) -> Tuple[int, np.ndarray]:
    """Classify video by averaging CLIP predictions"""
    text_prompts = [f"a video of {concept}" for concept in SEED_DV_CONCEPTS]
    
    all_probs = []
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
    
    avg_probs = np.mean(all_probs, axis=0)
    return int(np.argmax(avg_probs)), avg_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1-dir', type=str, required=True,
                       help='Output directory from stage 1')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: stage1-dir with _svd suffix)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-samples', type=int, default=None)
    # SVD settings
    parser.add_argument('--num-frames', type=int, default=14)
    parser.add_argument('--svd-steps', type=int, default=25)
    parser.add_argument('--motion-bucket', type=int, default=127)
    parser.add_argument('--use-blurry', action='store_true',
                       help='Use blurry frames instead of refined (for ablation)')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    stage1_dir = Path(args.stage1_dir)
    
    # Load stage 1 metadata
    with open(stage1_dir / "metadata.json") as f:
        stage1_data = json.load(f)
    
    samples = stage1_data['samples']
    settings = stage1_data['settings']
    
    print(f"Stage 2: SVD Video Generation")
    print(f"  Stage 1 dir: {stage1_dir}")
    print(f"  Samples: {len(samples)}")
    print(f"  Using: {'blurry' if args.use_blurry else 'refined'} frames")
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = "_svd_blurry" if args.use_blurry else "_svd"
        output_dir = stage1_dir.parent / f"{stage1_dir.name}{suffix}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "videos_pred").mkdir(exist_ok=True)
    (output_dir / "videos_gt").mkdir(exist_ok=True)
    (output_dir / "grids_pred").mkdir(exist_ok=True)
    (output_dir / "grids_gt").mkdir(exist_ok=True)
    (output_dir / "comparisons").mkdir(exist_ok=True)
    
    # Load models
    print("\nLoading models...")
    svd_pipe = load_svd_pipeline(device)
    clip_model, clip_processor = load_clip_classifier(device)
    
    print(f"\n{'='*60}")
    print(f"SVD Settings:")
    print(f"  Frames: {args.num_frames}")
    print(f"  Steps: {args.svd_steps}")
    print(f"  Motion bucket: {args.motion_bucket}")
    print(f"{'='*60}\n")
    
    # Process samples
    all_results = []
    correct_40way = 0
    correct_2way = 0
    correct_top5 = 0
    total_samples = 0
    
    max_samples = args.max_samples or len(samples)
    
    for sample in tqdm(samples[:max_samples]):
        idx = sample['sample_idx']
        concept_id = sample['concept_id']
        
        # Load first frame
        if args.use_blurry:
            frame_path = stage1_dir / "blurry" / f"{idx:05d}.png"
        else:
            frame_path = stage1_dir / "refined" / f"{idx:05d}.png"
        
        if not frame_path.exists():
            print(f"Warning: Frame {frame_path} not found, skipping")
            continue
        
        first_frame = np.array(Image.open(frame_path))
        
        # Load GT frames
        gt_dir = stage1_dir / "gt_frames" / f"{idx:05d}"
        gt_frames = load_gt_frames(gt_dir)
        
        # Generate video with SVD
        try:
            pred_frames = generate_video_svd(
                svd_pipe,
                first_frame,
                num_frames=args.num_frames,
                motion_bucket_id=args.motion_bucket,
                num_inference_steps=args.svd_steps,
            )
        except Exception as e:
            print(f"SVD failed for sample {idx}: {e}")
            pred_frames = [first_frame] * args.num_frames
        
        # Compute metrics
        video_ssim = compute_video_ssim(pred_frames, gt_frames)
        
        # Classification
        pred_class, pred_probs = classify_video_clip(
            pred_frames, clip_model, clip_processor, device
        )
        
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
            'sample_idx': idx,
            'subject_id': sample['subject_id'],
            'concept_id': concept_id,
            'concept_name': SEED_DV_CONCEPTS[concept_id],
            'pred_class': pred_class,
            'pred_name': SEED_DV_CONCEPTS[pred_class],
            'video_ssim': video_ssim,
            'correct_40way': bool(pred_class == concept_id),
            'correct_top5': bool(concept_id in top5_preds),
            'correct_2way': bool(gt_is_animal == pred_is_animal),
        }
        all_results.append(result)
        
        # Save outputs
        if idx < 50:
            save_video(pred_frames, output_dir / "videos_pred" / f"{idx:05d}.mp4")
            save_video(gt_frames, output_dir / "videos_gt" / f"{idx:05d}.mp4")
            save_frames_grid(pred_frames, output_dir / "grids_pred" / f"{idx:05d}.png")
            save_frames_grid(gt_frames, output_dir / "grids_gt" / f"{idx:05d}.png")
            
            # Comparison: first frame | middle pred | middle GT
            h, w = gt_frames[0].shape[:2]
            first_resized = cv2.resize(first_frame, (w, h))
            mid_pred = cv2.resize(pred_frames[len(pred_frames)//2], (w, h))
            mid_gt = gt_frames[len(gt_frames)//2]
            comparison = np.hstack([first_resized, mid_pred, mid_gt])
            Image.fromarray(comparison).save(output_dir / "comparisons" / f"{idx:05d}.png")
        
        # Progress
        if (total_samples) % 10 == 0:
            print(f"  Processed {total_samples}/{max_samples}, "
                  f"40-way: {correct_40way/total_samples*100:.1f}%, "
                  f"SSIM: {np.mean([r['video_ssim'] for r in all_results]):.3f}")
    
    # Summary
    ssim_values = [r['video_ssim'] for r in all_results]
    
    summary = {
        'total_samples': total_samples,
        'stage1_settings': settings,
        'svd_settings': {
            'num_frames': args.num_frames,
            'svd_steps': args.svd_steps,
            'motion_bucket': args.motion_bucket,
            'used_blurry': args.use_blurry,
        },
        'metrics': {
            'video_ssim_mean': float(np.mean(ssim_values)),
            'video_ssim_std': float(np.std(ssim_values)),
            'acc_40way': float(correct_40way / total_samples),
            'acc_top5': float(correct_top5 / total_samples),
            'acc_2way': float(correct_2way / total_samples),
        },
    }
    
    # Print results
    frame_type = "blurry" if args.use_blurry else "refined"
    print("\n" + "="*60)
    print(f"STAGE 2 RESULTS ({frame_type} → SVD)")
    print("="*60)
    print(f"\nTotal samples: {total_samples}")
    print(f"\nVideo-level metrics:")
    print(f"  Video SSIM: {summary['metrics']['video_ssim_mean']:.3f} ± {summary['metrics']['video_ssim_std']:.3f}")
    print(f"\nSemantic-level metrics:")
    print(f"  40-way accuracy: {summary['metrics']['acc_40way']*100:.1f}%")
    print(f"  Top-5 accuracy: {summary['metrics']['acc_top5']*100:.1f}%")
    print(f"  2-way accuracy: {summary['metrics']['acc_2way']*100:.1f}%")
    
    print("\n" + "="*60)
    print("COMPARISON WITH EEG2Video (NeurIPS 2024)")
    print("="*60)
    print(f"{'Metric':<20} {'EEG2Video':<15} {'Ours':<20}")
    print("-"*55)
    print(f"{'SSIM':<20} {'0.256 ± 0.03':<15} {summary['metrics']['video_ssim_mean']:.3f} ± {summary['metrics']['video_ssim_std']:.2f}")
    print(f"{'40-way acc':<20} {'15.9%':<15} {summary['metrics']['acc_40way']*100:.1f}%")
    print(f"{'2-way acc':<20} {'79.8%':<15} {summary['metrics']['acc_2way']*100:.1f}%")
    
    # Save
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
