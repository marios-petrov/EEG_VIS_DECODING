#!/usr/bin/env python3
"""
Stage 1 with Semantic Predictor: Fair EEG2Video comparison.

Uses PREDICTED concept (from semantic predictor) as SD3 prompt,
not GT captions. This matches EEG2Video's approach.

Pipeline:
    EEG → Semantic Predictor → Predicted CLIP embedding
                                        ↓
                            argmax similarity with 40 concepts
                                        ↓
                            Predicted concept name
                                        ↓
    EEG → EEGMamba → Blurry → SD3 (with predicted prompt) → Refined

Usage:
    python eval_stage1_refine_semantic.py \
        --checkpoint /path/to/eegmamba.pt \
        --semantic-predictor /path/to/semantic_predictor.pt \
        --classifier-dir /path/to/clip_classifier \
        --preprocessed-dir /path/to/preprocessed \
        --output-dir /path/to/output
"""

import os
os.environ['HF_HOME'] = '/local-scratch/marios-datasets/hf-cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/local-scratch/marios-datasets/hf-cache/hub'

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from typing import List, Dict

import sys
sys.path.append('/home/mpetrov/EEG_Reconstruction')


# SEED-DV Concepts (1-indexed)
SEED_DV_CONCEPTS = {
    1: "cat", 2: "dog", 3: "elephant", 4: "horse", 5: "panda",
    6: "rabbit", 7: "bird", 8: "fish", 9: "jellyfish", 10: "shark",
    11: "turtle", 12: "flower", 13: "mushroom", 14: "tree", 15: "boxing",
    16: "dancing", 17: "running", 18: "skiing", 19: "couple", 20: "face",
    21: "crowd", 22: "beach", 23: "buildings", 24: "mountain", 25: "road",
    26: "water", 27: "fireworks", 28: "banana", 29: "cake", 30: "drink",
    31: "pizza", 32: "watermelon", 33: "drum", 34: "guitar", 35: "piano",
    36: "bike", 37: "car", 38: "hot air balloon", 39: "airplane", 40: "ship",
}

def get_concept_name(concept_id):
    """Get concept name from ID (handles both 0-indexed and 1-indexed)"""
    if concept_id in SEED_DV_CONCEPTS:
        return SEED_DV_CONCEPTS[concept_id]
    elif concept_id + 1 in SEED_DV_CONCEPTS:
        return SEED_DV_CONCEPTS[concept_id + 1]
    return f"concept_{concept_id}"


class SemanticPredictor(nn.Module):
    """MLP to predict CLIP embeddings from EEG"""
    
    def __init__(
        self,
        eeg_channels: int = 62,
        eeg_samples: int = 400,
        hidden_dim: int = 1024,
        output_dim: int = 768,
        num_layers: int = 4,
    ):
        super().__init__()
        
        input_dim = eeg_channels * eeg_samples
        
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
        
        self.encoder = nn.Sequential(*layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        B = eeg.shape[0]
        x = eeg.view(B, -1)
        x = self.encoder(x)
        x = self.output_proj(x)
        return x


class SEEDDVDataset(Dataset):
    """SEED-DV test dataset"""
    
    def __init__(
        self,
        preprocessed_dir: Path,
        subject_ids: List[int],
        video_labels_path: str = None,
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.samples = []
        self.eeg_data = {}
        self.latent_data = {}
        
        # Load video labels for correct concept_id
        self.video_labels = None
        if video_labels_path and Path(video_labels_path).exists():
            self.video_labels = np.load(video_labels_path)
            print(f"  Loaded video labels: {self.video_labels.shape}")
        else:
            default_path = Path("/local-scratch/SEED/Video/meta-info/All_video_label.npy")
            if default_path.exists():
                self.video_labels = np.load(default_path)
                print(f"  Loaded video labels from default: {self.video_labels.shape}")
        
        for subj_id in subject_ids:
            subj_dir = self.preprocessed_dir / f"sub{subj_id}"
            if not subj_dir.exists():
                continue
            
            eeg_path = subj_dir / "eeg.npy"
            latent_path = subj_dir / "latents.npy"
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
                    
                    # Fix concept_id
                    if self.video_labels is not None:
                        block_id = seg['block_id']
                        position = seg['concept_id']
                        if 0 <= block_id < 7 and 0 <= position < 40:
                            seg['concept_id'] = int(self.video_labels[block_id, position])
                    
                    self.samples.append(seg)
        
        print(f"  Loaded {len(self.samples)} test samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        seg = self.samples[idx]
        subj_id = seg['subject_id']
        seg_idx = seg['segment_idx']
        
        eeg = self.eeg_data[subj_id][seg_idx]
        all_latents = self.latent_data[subj_id][seg_idx]
        
        return {
            'eeg': torch.from_numpy(eeg).float(),
            'all_latents': torch.from_numpy(all_latents).float(),
            'subject_id': subj_id,
            'concept_id': seg['concept_id'],
            'segment_idx': seg_idx,
        }


def load_vae():
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    vae.eval()
    return vae


def decode_latent(latent: torch.Tensor, vae, device) -> np.ndarray:
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


def predict_concept(eeg: torch.Tensor, semantic_predictor, text_features, device) -> int:
    """Predict concept from EEG using semantic predictor"""
    with torch.no_grad():
        # Get predicted CLIP embedding
        pred_emb = semantic_predictor(eeg.to(device))
        pred_emb = F.normalize(pred_emb, dim=-1)
        
        # Compare with concept text features
        text_features = text_features.to(device).float()
        text_features = F.normalize(text_features, dim=-1)
        
        similarities = pred_emb @ text_features.T  # [B, 40]
        pred_idx = similarities.argmax(dim=-1).item()  # 0-indexed
    
    return pred_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='EEGMamba adapter checkpoint')
    parser.add_argument('--semantic-predictor', type=str, required=True,
                       help='Semantic predictor checkpoint')
    parser.add_argument('--classifier-dir', type=str, required=True,
                       help='CLIP classifier directory (with text_features.pt)')
    parser.add_argument('--preprocessed-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--subjects', type=str, default='1-20')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--video-labels', type=str,
                       default='/local-scratch/SEED/Video/meta-info/All_video_label.npy')
    # SD3 settings
    parser.add_argument('--strength', type=float, default=0.5)
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
    
    print(f"{'='*60}")
    print("STAGE 1 WITH SEMANTIC PREDICTOR (Fair EEG2Video Comparison)")
    print(f"{'='*60}")
    print(f"Subjects: {subject_ids}")
    
    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "blurry").mkdir(exist_ok=True)
    (output_dir / "refined").mkdir(exist_ok=True)
    
    # Load models
    print("\nLoading models...")
    
    # VAE
    vae = load_vae().to(device)
    print("  ✓ VAE loaded")
    
    # SD3 img2img
    from diffusers import StableDiffusion3Img2ImgPipeline
    sd3_pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.float16,
    ).to(device)
    sd3_pipe.set_progress_bar_config(disable=True)
    print("  ✓ SD3 pipeline loaded")
    
    # EEGMamba
    from eegmamba_adapter_optimized import EEGMambaAdapter
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    eegmamba = EEGMambaAdapter(
        eeg_channels=62,
        eeg_samples=400,
        d_model=512,
        n_layers=4,
        latent_channels=16,
        latent_height=96,
        latent_width=96,
        text_embed_dim=2048,
    ).to(device)
    eegmamba.load_state_dict(checkpoint['model'])
    eegmamba.eval()
    print(f"  ✓ EEGMamba loaded (epoch {checkpoint.get('epoch', '?')})")
    
    # Semantic predictor
    sem_checkpoint = torch.load(args.semantic_predictor, map_location=device, weights_only=False)
    config = sem_checkpoint['config']
    semantic_predictor = SemanticPredictor(
        eeg_channels=config['eeg_channels'],
        eeg_samples=config['eeg_samples'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        num_layers=config['num_layers'],
    ).to(device)
    semantic_predictor.load_state_dict(sem_checkpoint['model'])
    semantic_predictor.eval()
    print(f"  ✓ Semantic predictor loaded (acc: {sem_checkpoint.get('acc_40way', 0)*100:.1f}%)")
    
    # Load concept text features
    text_features_path = Path(args.classifier_dir) / "text_features.pt"
    text_features = torch.load(text_features_path)  # [40, 768]
    print(f"  ✓ Text features loaded: {text_features.shape}")
    
    # Create dataset
    test_dataset = SEEDDVDataset(
        Path(args.preprocessed_dir),
        subject_ids,
        video_labels_path=args.video_labels,
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"\n{'='*60}")
    print(f"Settings:")
    print(f"  Strength: {args.strength}")
    print(f"  Steps: {args.num_inference_steps}")
    print(f"  Guidance: {args.guidance_scale}")
    print(f"  Prompt source: PREDICTED (semantic predictor)")
    print(f"{'='*60}\n")
    
    # Process
    metadata_list = []
    correct_predictions = 0
    max_samples = args.max_samples or len(test_dataset)
    
    for batch_idx, batch in enumerate(tqdm(test_loader, total=min(max_samples, len(test_loader)))):
        if batch_idx >= max_samples:
            break
        
        eeg = batch['eeg'].to(device)
        all_latents = batch['all_latents'][0]
        gt_concept_id = batch['concept_id'].item()
        subject_id = batch['subject_id'].item()
        
        # Predict concept using semantic predictor
        pred_idx = predict_concept(eeg, semantic_predictor, text_features, device)
        pred_concept_id = pred_idx + 1  # Convert to 1-indexed
        pred_concept_name = get_concept_name(pred_concept_id)
        gt_concept_name = get_concept_name(gt_concept_id)
        
        # Track accuracy
        gt_idx = gt_concept_id - 1 if gt_concept_id >= 1 else gt_concept_id
        if pred_idx == gt_idx:
            correct_predictions += 1
        
        # Generate blurry prediction from EEG
        with torch.no_grad():
            caption_emb = torch.zeros(1, 2048).to(device)  # No caption conditioning for EEGMamba
            with torch.cuda.amp.autocast():
                pred_latent = eegmamba(eeg, caption_emb)
        
        blurry_img = decode_latent(pred_latent[0], vae, device)
        
        # Refine with SD3 using PREDICTED concept as prompt
        prompt = f"a video of a {pred_concept_name}"
        pil_blurry = Image.fromarray(blurry_img).convert("RGB")
        
        with torch.no_grad():
            refined_result = sd3_pipe(
                prompt=prompt,
                image=pil_blurry,
                strength=args.strength,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            ).images[0]
        
        refined_img = np.array(refined_result)
        
        # Save
        Image.fromarray(blurry_img).save(output_dir / "blurry" / f"{batch_idx:05d}.png")
        Image.fromarray(refined_img).save(output_dir / "refined" / f"{batch_idx:05d}.png")
        
        metadata_list.append({
            'sample_idx': batch_idx,
            'subject_id': subject_id,
            'gt_concept_id': gt_concept_id,
            'gt_concept_name': gt_concept_name,
            'pred_concept_id': pred_concept_id,
            'pred_concept_name': pred_concept_name,
            'concept_id': gt_concept_id,  # For compatibility with eval scripts
            'prompt_used': prompt,
            'correct_prediction': pred_idx == gt_idx,
        })
        
        # Progress
        if (batch_idx + 1) % 100 == 0:
            acc = correct_predictions / (batch_idx + 1)
            print(f"  Processed {batch_idx+1}, Semantic acc: {acc*100:.1f}%")
    
    # Save metadata
    final_acc = correct_predictions / len(metadata_list) if metadata_list else 0
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump({
            'settings': {
                'strength': args.strength,
                'num_inference_steps': args.num_inference_steps,
                'guidance_scale': args.guidance_scale,
                'prompt_source': 'semantic_predictor',
                'semantic_predictor_acc': final_acc,
            },
            'samples': metadata_list,
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("STAGE 1 COMPLETE (Semantic Predictor)")
    print(f"{'='*60}")
    print(f"Processed: {len(metadata_list)} samples")
    print(f"Semantic predictor accuracy: {final_acc*100:.1f}%")
    print(f"Output: {output_dir}")
    print(f"\nNext: Run classification evaluation:")
    print(f"  python eval_seed_dv_classification.py \\")
    print(f"      --classifier-dir {args.classifier_dir} \\")
    print(f"      --images-dir {output_dir}/refined \\")
    print(f"      --metadata {output_dir}/metadata.json")


if __name__ == "__main__":
    main()
