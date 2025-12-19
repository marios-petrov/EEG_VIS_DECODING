#!/usr/bin/env python3
"""
SEED-DV Classification Evaluation

Loads precomputed CLIP classifier and evaluates generated images.
Run prepare_clip_classifier.py first to create the classifier.

Metrics:
- 40-way Top-1 Accuracy (paper: 13.8% frame, 15.9% video)  
- 2-way Animal/Non-animal (paper: 77.4% frame, 79.8% video)

Usage:
    python eval_seed_dv_classification.py \
        --classifier-dir /local-scratch/marios-datasets/SEED/clip_classifier \
        --images-dir /path/to/stage1_results/refined \
        --metadata /path/to/stage1_results/metadata.json
"""

import os
os.environ['HF_HOME'] = '/local-scratch/marios-datasets/hf-cache'

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
import random


def load_classifier(classifier_dir: Path, device):
    """Load precomputed CLIP classifier"""
    from transformers import CLIPProcessor, CLIPModel
    
    # Load text features
    features_path = classifier_dir / "text_features.pt"
    text_features = torch.load(features_path).to(device)
    
    # Load concept info
    info_path = classifier_dir / "concept_info.json"
    with open(info_path) as f:
        concept_info = json.load(f)
    
    # Load CLIP model for image encoding
    clip_model = concept_info.get("clip_model", "openai/clip-vit-large-patch14")
    
    processor_path = classifier_dir / "processor"
    if processor_path.exists():
        processor = CLIPProcessor.from_pretrained(processor_path)
    else:
        processor = CLIPProcessor.from_pretrained(clip_model)
    
    model = CLIPModel.from_pretrained(
        clip_model,
        torch_dtype=torch.float16,
    ).to(device).eval()
    
    return model, processor, text_features, concept_info


def classify_image(image_path: Path, model, processor, text_features, device):
    """Classify a single image, returns probs and predicted index (0-indexed)"""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # Cosine similarity with temperature scaling
    similarities = (image_features @ text_features.T)[0]
    probs = F.softmax(similarities * 100, dim=0).cpu().numpy()
    pred_idx = int(similarities.argmax().item())
    
    return probs, pred_idx


def n_way_accuracy(probs: np.ndarray, gt_idx: int, n_way: int = 40, top_k: int = 1) -> float:
    """N-way top-K accuracy"""
    if n_way >= len(probs):
        top_indices = np.argsort(probs)[-top_k:]
        return float(gt_idx in top_indices)
    
    # Sample distractors
    others = [i for i in range(len(probs)) if i != gt_idx]
    distractors = random.sample(others, n_way - 1)
    selected = distractors + [gt_idx]
    
    selected_probs = probs[selected]
    gt_pos = selected.index(gt_idx)
    top_pos = np.argsort(selected_probs)[-top_k:]
    
    return float(gt_pos in top_pos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier-dir', type=str, required=True,
                       help='Directory with precomputed classifier (from prepare_clip_classifier.py)')
    parser.add_argument('--images-dir', type=str, required=True,
                       help='Directory with generated images')
    parser.add_argument('--metadata', type=str, required=True,
                       help='Path to metadata.json with sample info')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-samples', type=int, default=None)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    classifier_dir = Path(args.classifier_dir)
    images_dir = Path(args.images_dir)
    
    # Load metadata
    with open(args.metadata) as f:
        metadata = json.load(f)
    samples = metadata['samples']
    
    print(f"{'='*60}")
    print("SEED-DV CLASSIFICATION EVALUATION")
    print(f"{'='*60}")
    print(f"Classifier: {classifier_dir}")
    print(f"Images: {images_dir}")
    print(f"Samples: {len(samples)}")
    
    # Load classifier
    print("\nLoading classifier...")
    model, processor, text_features, concept_info = load_classifier(classifier_dir, device)
    concepts = concept_info['concepts']
    n_animals = concept_info['n_animals']
    print(f"  ✓ Loaded ({text_features.shape[0]} concepts, {text_features.shape[1]}D)")
    
    # Detect indexing from metadata
    sample_ids = [s['concept_id'] for s in samples[:50]]
    is_1indexed = min(sample_ids) >= 1 and max(sample_ids) <= 40
    print(f"  Concept IDs: {'1-indexed' if is_1indexed else '0-indexed'}")
    
    # Output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = images_dir.parent / "eval_classification"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate
    results = []
    correct_40way = 0
    correct_2way = 0
    total = 0
    
    concept_stats = defaultdict(lambda: {"total": 0, "correct_40": 0, "correct_2": 0})
    
    max_samples = args.max_samples or len(samples)
    
    print(f"\nEvaluating {min(max_samples, len(samples))} images...")
    
    for sample in tqdm(samples[:max_samples]):
        idx = sample['sample_idx']
        concept_id_raw = sample['concept_id']
        
        # Convert to 0-indexed
        gt_idx = concept_id_raw - 1 if is_1indexed else concept_id_raw
        
        if gt_idx < 0 or gt_idx >= 40:
            continue
        
        # Find image
        image_path = images_dir / f"{idx:05d}.png"
        if not image_path.exists():
            image_path = images_dir / f"{idx}.png"
        if not image_path.exists():
            continue
        
        # Classify
        probs, pred_idx = classify_image(image_path, model, processor, text_features, device)
        
        # 40-way accuracy
        acc_40 = n_way_accuracy(probs, gt_idx, n_way=40, top_k=1)
        correct_40way += acc_40
        
        # 2-way accuracy (animal vs non-animal)
        gt_animal = gt_idx < n_animals
        pred_animal = pred_idx < n_animals
        acc_2 = int(gt_animal == pred_animal)
        correct_2way += acc_2
        
        total += 1
        
        # Stats
        concept_stats[gt_idx]["total"] += 1
        concept_stats[gt_idx]["correct_40"] += acc_40
        concept_stats[gt_idx]["correct_2"] += acc_2
        
        results.append({
            'sample_idx': idx,
            'gt_idx': gt_idx,
            'gt_name': concepts[gt_idx]['name'],
            'pred_idx': pred_idx,
            'pred_name': concepts[pred_idx]['name'],
            'correct_40way': acc_40,
            'correct_2way': acc_2,
        })
    
    # Compute final metrics
    acc_40way = correct_40way / total if total > 0 else 0
    acc_2way = correct_2way / total if total > 0 else 0
    
    # Baselines
    random_40 = 1 / 40
    p_animal = n_animals / 40
    random_2way = p_animal**2 + (1 - p_animal)**2
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total evaluated: {total}")
    print(f"\n40-way Top-1: {acc_40way*100:.1f}%  (random: {random_40*100:.1f}%)")
    print(f"2-way:        {acc_2way*100:.1f}%  (random: {random_2way*100:.1f}%)")
    
    print(f"\n{'='*60}")
    print("COMPARISON WITH EEG2Video PAPER")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Paper':<12} {'Ours':<12} {'Δ'}")
    print("-"*50)
    print(f"{'40-way (frame)':<20} {'13.8%':<12} {acc_40way*100:.1f}%{'':<6} {acc_40way*100-13.8:+.1f}%")
    print(f"{'2-way (frame)':<20} {'77.4%':<12} {acc_2way*100:.1f}%{'':<6} {acc_2way*100-77.4:+.1f}%")
    
    # Per-concept breakdown (top/bottom 5)
    concept_accs = [(idx, s["correct_40"]/s["total"], s["total"]) 
                    for idx, s in concept_stats.items() if s["total"] > 0]
    concept_accs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*60}")
    print("TOP 5 CONCEPTS")
    print(f"{'='*60}")
    for idx, acc, n in concept_accs[:5]:
        print(f"  {concepts[idx]['name']:<20} {acc*100:.1f}% (n={n})")
    
    print(f"\nBOTTOM 5 CONCEPTS")
    for idx, acc, n in concept_accs[-5:]:
        print(f"  {concepts[idx]['name']:<20} {acc*100:.1f}% (n={n})")
    
    # Save results
    summary = {
        'metrics': {
            'acc_40way': acc_40way,
            'acc_2way': acc_2way,
            'random_40way': random_40,
            'random_2way': random_2way,
        },
        'paper': {
            'frame_40way': 0.138,
            'frame_2way': 0.774,
        },
        'total_evaluated': total,
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
