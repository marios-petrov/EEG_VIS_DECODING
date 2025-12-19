#!/usr/bin/env python3
"""
Utility functions for EEG2Video
- Data loading and preprocessing
- Metrics (SSIM, PSNR, LPIPS, optical flow)
- Video I/O and processing
- Visualization helpers
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import cv2
from tqdm import tqdm

# Dataset classes
from torch.utils.data import Dataset


class EEGVideoDataset(Dataset):
    """Dataset for EEG-video pairs with synchronized preprocessing"""
    
    def __init__(
        self,
        preprocessed_dir: Path,
        task: str,
        split: str = "train",
        subjects: Optional[List[str]] = None,
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.task = task
        self.split = split
        
        # Load metadata
        meta_path = self.preprocessed_dir / f"{task}_metadata.json"
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load EEG data
        eeg_path = self.preprocessed_dir / f"{task}_eeg.npy"
        self.eeg_data = np.load(eeg_path)  # [num_segments, channels, time_steps]
        
        # Load video latents
        latents_path = self.preprocessed_dir / f"{task}_vae_latents_hd.npy"
        self.latents = np.load(latents_path)  # [num_segments, frames, C, H, W]
        
        # Load subject IDs if available
        subject_path = self.preprocessed_dir / f"{task}_subject_ids.npy"
        if subject_path.exists():
            self.subject_ids = np.load(subject_path)
        else:
            self.subject_ids = np.zeros(len(self.eeg_data), dtype=np.int64)
        
        # Filter by subjects if specified
        if subjects is not None:
            subject_mapping = self.metadata.get("subject_mapping", {})
            subject_indices = [int(subject_mapping.get(s, -1)) for s in subjects]
            mask = np.isin(self.subject_ids, subject_indices)
            self.eeg_data = self.eeg_data[mask]
            self.latents = self.latents[mask]
            self.subject_ids = self.subject_ids[mask]
        
        # Split data (80/10/10 train/val/test)
        n = len(self.eeg_data)
        if split == "train":
            self.indices = range(0, int(0.8 * n))
        elif split == "val":
            self.indices = range(int(0.8 * n), int(0.9 * n))
        else:  # test
            self.indices = range(int(0.9 * n), n)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = list(self.indices)[idx]
        eeg = torch.from_numpy(self.eeg_data[actual_idx]).float()
        latent = torch.from_numpy(self.latents[actual_idx]).float()
        subject_id = torch.tensor(self.subject_ids[actual_idx], dtype=torch.long)
        
        return {
            'eeg': eeg,
            'latent': latent,
            'subject_id': subject_id,
            'index': actual_idx,
        }


class LatentFramesWithPrompts(Dataset):
    """Dataset for fine-tuning UNet with latents and text prompts"""
    
    def __init__(
        self,
        preprocessed_dir: Path,
        task: str,
        style_suffix: str = "animated movie still, cinematic lighting, high detail",
        flow_weighting: bool = True,
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.task = task
        
        # Load latents
        latents = np.load(preprocessed_dir / f"{task}_vae_latents_hd.npy")
        S, F = latents.shape[0], latents.shape[1]
        self.latents = torch.from_numpy(latents).float().view(-1, 4, 96, 96)
        
        # Load captions
        captions_path = preprocessed_dir / f"{task}_captions_hd.json"
        if captions_path.exists():
            with open(captions_path, "r") as f:
                data = json.load(f)
                caps = data.get("captions", [])
        else:
            caps = []
        
        if not caps or len(caps) != S:
            caps = [""] * S
        
        # Load optical flow scores for dynamic motion prompts
        flows = None
        flow_path = preprocessed_dir / f"{task}_flow_scores_hd.npy"
        if flow_path.exists() and flow_weighting:
            flows = np.load(flow_path)
            if flows.size > 0:
                fmin, fmax = float(flows.min()), float(flows.max())
                flows = (flows - fmin) / (fmax - fmin) if fmax > fmin else np.zeros_like(flows)
        
        # Build prompts
        seg_prompts = []
        for si in range(S):
            cap = (caps[si] or "").strip()
            base = f"{cap}. {style_suffix}" if cap else style_suffix
            
            # Add motion prompt for high-flow segments
            if flows is not None and flows.size == S and flows[si] > 0.66:
                base = base + ", dynamic motion, action shot"
            
            seg_prompts.append(base)
        
        # Expand prompts for all frames
        self.prompts = []
        for si in range(S):
            self.prompts.extend([seg_prompts[si]] * F)
        
        assert len(self.prompts) == self.latents.shape[0]
    
    def __len__(self):
        return self.latents.size(0)
    
    def __getitem__(self, idx):
        return self.latents[idx], self.prompts[idx]


# Video processing utilities

def read_video_at_fps(video_path: Path, target_fps: int) -> List[np.ndarray]:
    """Read video frames at target FPS"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    keep_every = max(1, int(round(src_fps / target_fps)))
    
    frames, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % keep_every == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    
    return frames


def segment_frames(frames: List, frames_per_seg: int, hop: int) -> List[List]:
    """Segment frames into overlapping windows"""
    segments = []
    for i in range(0, len(frames) - frames_per_seg + 1, hop):
        segments.append(frames[i:i + frames_per_seg])
    return segments


def square_resize(img_array: np.ndarray, size: int) -> Image.Image:
    """Center crop to square and resize"""
    img = Image.fromarray(img_array)
    w, h = img.size
    s = min(w, h)
    l = (w - s) // 2
    t = (h - s) // 2
    cropped = img.crop((l, t, l + s, t + s))
    return cropped.resize((size, size), Image.BILINEAR)


def frames_to_video(
    frames: List[np.ndarray],
    output_path: Path,
    fps: int = 6,
    codec: str = 'mp4v'
) -> None:
    """Save frames as video file"""
    if not frames:
        raise ValueError("No frames to save")
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)
    
    writer.release()


def create_side_by_side_video(
    gt_frames: List[np.ndarray],
    pred_frames: List[np.ndarray],
    output_path: Path,
    fps: int = 6,
    labels: Tuple[str, str] = ("Ground Truth", "Predicted"),
) -> None:
    """Create side-by-side comparison video"""
    if len(gt_frames) != len(pred_frames):
        raise ValueError("Frame counts don't match")
    
    # Add text labels
    labeled_gt = []
    labeled_pred = []
    
    for gt, pred in zip(gt_frames, pred_frames):
        gt_labeled = gt.copy()
        pred_labeled = pred.copy()
        
        # Add text
        cv2.putText(gt_labeled, labels[0], (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(pred_labeled, labels[1], (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        labeled_gt.append(gt_labeled)
        labeled_pred.append(pred_labeled)
    
    # Concatenate side by side
    combined_frames = [
        np.hstack([gt, pred]) 
        for gt, pred in zip(labeled_gt, labeled_pred)
    ]
    
    frames_to_video(combined_frames, output_path, fps)


# Metric calculations

def calculate_ssim_batch(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> torch.Tensor:
    """
    Calculate SSIM between batches of images
    Args:
        img1, img2: [B, C, H, W] tensors in range [0, 1]
    Returns:
        ssim: scalar or [B] tensor
    """
    from pytorch_msssim import ssim
    return ssim(img1, img2, window_size=window_size, size_average=size_average)


def calculate_psnr_batch(
    img1: torch.Tensor,
    img2: torch.Tensor,
    max_val: float = 1.0,
) -> torch.Tensor:
    """
    Calculate PSNR between batches of images
    Args:
        img1, img2: [B, C, H, W] tensors
        max_val: Maximum pixel value
    Returns:
        psnr: [B] tensor
    """
    mse = F.mse_loss(img1, img2, reduction='none').mean(dim=[1, 2, 3])
    psnr = 10 * torch.log10(max_val**2 / (mse + 1e-10))
    return psnr


def calculate_lpips_batch(
    img1: torch.Tensor,
    img2: torch.Tensor,
    lpips_model,
) -> torch.Tensor:
    """
    Calculate LPIPS perceptual distance
    Args:
        img1, img2: [B, C, H, W] tensors in range [-1, 1]
        lpips_model: Pretrained LPIPS model
    Returns:
        lpips: [B] tensor
    """
    with torch.no_grad():
        lpips_scores = lpips_model(img1, img2)
    return lpips_scores.squeeze()


def calculate_optical_flow(
    frames: List[np.ndarray],
    method: str = 'farneback'
) -> List[np.ndarray]:
    """
    Calculate optical flow between consecutive frames
    Args:
        frames: List of RGB frames [H, W, 3]
        method: 'farneback' or 'lucaskanade'
    Returns:
        flows: List of flow fields [H, W, 2]
    """
    if len(frames) < 2:
        return []
    
    flows = []
    for i in range(len(frames) - 1):
        prev = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        next = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
        
        if method == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(
                prev, next, None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
        else:
            raise ValueError(f"Unknown flow method: {method}")
        
        flows.append(flow)
    
    return flows


def flow_magnitude(flow: np.ndarray) -> float:
    """Calculate average flow magnitude"""
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return float(mag.mean())


def calculate_flow_metrics(
    gt_frames: List[np.ndarray],
    pred_frames: List[np.ndarray],
) -> Dict[str, float]:
    """
    Calculate flow-based temporal consistency metrics
    Returns:
        Dict with 'flow_mse', 'flow_corr', 'avg_flow_gt', 'avg_flow_pred'
    """
    gt_flows = calculate_optical_flow(gt_frames)
    pred_flows = calculate_optical_flow(pred_frames)
    
    if not gt_flows or not pred_flows:
        return {
            'flow_mse': float('nan'),
            'flow_corr': float('nan'),
            'avg_flow_gt': float('nan'),
            'avg_flow_pred': float('nan'),
        }
    
    # Calculate MSE between flows
    flow_mse = np.mean([
        np.mean((gt - pred)**2) 
        for gt, pred in zip(gt_flows, pred_flows)
    ])
    
    # Calculate correlation
    gt_mags = [flow_magnitude(f) for f in gt_flows]
    pred_mags = [flow_magnitude(f) for f in pred_flows]
    flow_corr = np.corrcoef(gt_mags, pred_mags)[0, 1]
    
    return {
        'flow_mse': float(flow_mse),
        'flow_corr': float(flow_corr),
        'avg_flow_gt': float(np.mean(gt_mags)),
        'avg_flow_pred': float(np.mean(pred_mags)),
    }


# General utilities

def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict] = None,
):
    """Save training checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device('cuda'),
) -> Dict:
    """Load training checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Pretty print metrics"""
    if prefix:
        print(f"\n{prefix}")
    print("-" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.4f}")
        else:
            print(f"  {key:20s}: {value}")
    print("-" * 60)


def normalize_tensor(x: torch.Tensor, min_val: float = -1, max_val: float = 1) -> torch.Tensor:
    """Normalize tensor to [min_val, max_val]"""
    x_min = x.min()
    x_max = x.max()
    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)
        x = x * (max_val - min_val) + min_val
    return x


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor [C, H, W] in [-1, 1] to PIL Image"""
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and PIL
    array = tensor.cpu().numpy().transpose(1, 2, 0)
    array = (array * 255).astype(np.uint8)
    return Image.fromarray(array)


def pil_to_tensor(image: Image.Image, normalize: bool = True) -> torch.Tensor:
    """Convert PIL Image to tensor [C, H, W]"""
    array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    
    if normalize:
        # Normalize to [-1, 1]
        tensor = tensor * 2 - 1
    
    return tensor
