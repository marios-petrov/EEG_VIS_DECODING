#!/usr/bin/env python3
"""
Proper evaluation matching EEG2Video paper methodology.

Key differences from our previous eval:
1. N-way classification: Sample N-1 distractors + GT, not all 40 classes
2. Evaluate FIRST frame (EEG prediction), not SVD-generated frames
3. Use multiple CLIP prompt templates like standard benchmarks
4. Proper 2-way split based on actual concept categories

Usage:
    python eval_proper_classification.py \
        --stage1-dir /path/to/stage1_results \
        --output-dir /path/to/eval_results
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
from typing import List, Tuple, Dict
import random
from collections import defaultdict

# SEED-DV 40 concepts with category annotations
SEED_DV_CONCEPTS = [
    # 0-4: Aerial/Flight
    {"name": "hot air balloon", "category": "aerial", "has_animal": False},
    {"name": "roller coaster", "category": "ride", "has_animal": False},
    {"name": "drone racing", "category": "aerial", "has_animal": False},
    {"name": "wing suit flying", "category": "aerial", "has_animal": False},
    {"name": "ski lift", "category": "ride", "has_animal": False},
    # 5-9: Animals
    {"name": "dogs playing", "category": "animal", "has_animal": True},
    {"name": "cats playing", "category": "animal", "has_animal": True},
    {"name": "underwater diving", "category": "water", "has_animal": False},  # Could have fish but not primary
    {"name": "safari animals", "category": "animal", "has_animal": True},
    {"name": "horse riding", "category": "animal", "has_animal": True},
    # 10-14: Entertainment
    {"name": "fireworks display", "category": "entertainment", "has_animal": False},
    {"name": "concert performance", "category": "entertainment", "has_animal": False},
    {"name": "street dance", "category": "entertainment", "has_animal": False},
    {"name": "magic show", "category": "entertainment", "has_animal": False},
    {"name": "circus act", "category": "entertainment", "has_animal": False},
    # 15-19: Media/Shows
    {"name": "cooking show", "category": "media", "has_animal": False},
    {"name": "sports highlights", "category": "sports", "has_animal": False},
    {"name": "news broadcast", "category": "media", "has_animal": False},
    {"name": "weather forecast", "category": "media", "has_animal": False},
    {"name": "talk show", "category": "media", "has_animal": False},
    # 20-24: Vehicles
    {"name": "car racing", "category": "vehicle", "has_animal": False},
    {"name": "motorcycle riding", "category": "vehicle", "has_animal": False},
    {"name": "boat sailing", "category": "vehicle", "has_animal": False},
    {"name": "train journey", "category": "vehicle", "has_animal": False},
    {"name": "airplane takeoff", "category": "vehicle", "has_animal": False},
    # 25-29: Extreme Sports
    {"name": "mountain climbing", "category": "extreme_sport", "has_animal": False},
    {"name": "surfing waves", "category": "extreme_sport", "has_animal": False},
    {"name": "skateboarding", "category": "extreme_sport", "has_animal": False},
    {"name": "parkour running", "category": "extreme_sport", "has_animal": False},
    {"name": "bungee jumping", "category": "extreme_sport", "has_animal": False},
    # 30-34: Art/Craft
    {"name": "painting art", "category": "art", "has_animal": False},
    {"name": "sculpture making", "category": "art", "has_animal": False},
    {"name": "pottery crafting", "category": "art", "has_animal": False},
    {"name": "glass blowing", "category": "art", "has_animal": False},
    {"name": "woodworking", "category": "art", "has_animal": False},
    # 35-39: Science/Nature
    {"name": "science experiment", "category": "science", "has_animal": False},
    {"name": "robot demonstration", "category": "science", "has_animal": False},
    {"name": "space footage", "category": "science", "has_animal": False},
    {"name": "nature documentary", "category": "nature", "has_animal": True},  # Often has animals
    {"name": "city timelapse", "category": "urban", "has_animal": False},
]

# Multiple prompt templates (like standard CLIP benchmarks)
CLIP_TEMPLATES = [
    "a photo of {}",
    "a video of {}",
    "a video showing {}",
    "a scene of {}",
    "{}",
]


def load_clip_model(device):
    """Load CLIP model"""
    from transformers import CLIPProcessor, CLIPModel
    
    print("Loading CLIP model...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        torch_dtype=torch.float16,
    ).to(device).eval()
    print("  ✓ CLIP loaded")
    
    return model, processor


def get_clip_text_features(model, processor, concepts: List[str], device) -> torch.Tensor:
    """Get CLIP text features for concepts using multiple templates"""
    all_features = []
    
    for concept in concepts:
        # Average features across templates
        template_features = []
        for template in CLIP_TEMPLATES:
            text = template.format(concept)
            inputs = processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                features = model.get_text_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
            
            template_features.append(features)
        
        # Average across templates
        avg_features = torch.stack(template_features).mean(dim=0)
        avg_features = avg_features / avg_features.norm(dim=-1, keepdim=True)
        all_features.append(avg_features)
    
    return torch.cat(all_features, dim=0)  # [N_concepts, D]


def get_clip_image_features(model, processor, image: np.ndarray, device) -> torch.Tensor:
    """Get CLIP image features"""
    pil_image = Image.fromarray(image)
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    
    return features  # [1, D]


def n_way_classification(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    gt_idx: int,
    n_way: int = 40,
    top_k: int = 1,
    n_trials: int = 10,
) -> Tuple[float, int]:
    """
    N-way top-K classification matching paper methodology.
    
    Args:
        image_features: [1, D] image features
        text_features: [40, D] all concept text features
        gt_idx: ground truth concept index
        n_way: number of classes to choose from (including GT)
        top_k: check if GT is in top-K predictions
        n_trials: number of random trials to average
    
    Returns:
        accuracy: fraction of trials where GT was in top-K
        best_pred: most common prediction across trials
    """
    n_concepts = text_features.shape[0]
    
    if n_way >= n_concepts:
        # Use all classes
        similarities = (image_features @ text_features.T)[0]  # [40]
        top_k_preds = similarities.topk(top_k).indices.cpu().numpy()
        return float(gt_idx in top_k_preds), int(similarities.argmax().item())
    
    # Multiple trials with random distractors
    successes = 0
    all_preds = []
    
    for _ in range(n_trials):
        # Sample N-1 distractors + GT
        other_indices = [i for i in range(n_concepts) if i != gt_idx]
        distractors = random.sample(other_indices, n_way - 1)
        selected_indices = distractors + [gt_idx]
        random.shuffle(selected_indices)
        
        # Get position of GT in selected
        gt_position = selected_indices.index(gt_idx)
        
        # Compute similarities for selected concepts only
        selected_features = text_features[selected_indices]
        similarities = (image_features @ selected_features.T)[0]
        
        # Check if GT is in top-K
        top_k_positions = similarities.topk(top_k).indices.cpu().numpy()
        if gt_position in top_k_positions:
            successes += 1
        
        # Track prediction (map back to original index)
        pred_position = similarities.argmax().item()
        pred_idx = selected_indices[pred_position]
        all_preds.append(pred_idx)
    
    # Most common prediction
    from collections import Counter
    best_pred = Counter(all_preds).most_common(1)[0][0]
    
    return successes / n_trials, best_pred


def two_way_classification(gt_idx: int, pred_idx: int, mode: str = "animal") -> bool:
    """
    2-way classification.
    
    Modes:
        - "animal": animal vs non-animal (based on has_animal flag)
        - "category": same broad category
    """
    if mode == "animal":
        gt_has_animal = SEED_DV_CONCEPTS[gt_idx]["has_animal"]
        pred_has_animal = SEED_DV_CONCEPTS[pred_idx]["has_animal"]
        return gt_has_animal == pred_has_animal
    elif mode == "category":
        gt_category = SEED_DV_CONCEPTS[gt_idx]["category"]
        pred_category = SEED_DV_CONCEPTS[pred_idx]["category"]
        return gt_category == pred_category
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1-dir', type=str, required=True,
                       help='Directory with refined/blurry frames from stage 1')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use-blurry', action='store_true',
                       help='Evaluate blurry frames instead of refined')
    parser.add_argument('--n-way', type=int, default=40,
                       help='N for N-way classification (40=all classes)')
    parser.add_argument('--top-k', type=int, default=1,
                       help='K for top-K accuracy')
    parser.add_argument('--n-trials', type=int, default=10,
                       help='Number of trials for N-way sampling')
    parser.add_argument('--max-samples', type=int, default=None)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    stage1_dir = Path(args.stage1_dir)
    
    # Load metadata
    with open(stage1_dir / "metadata.json") as f:
        stage1_data = json.load(f)
    
    samples = stage1_data['samples']
    settings = stage1_data['settings']
    
    print(f"\nProper Classification Evaluation")
    print(f"  Stage 1 dir: {stage1_dir}")
    print(f"  Samples: {len(samples)}")
    print(f"  Frame type: {'blurry' if args.use_blurry else 'refined'}")
    print(f"  N-way: {args.n_way}")
    print(f"  Top-K: {args.top_k}")
    
    # Setup output
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = "_blurry" if args.use_blurry else "_refined"
        output_dir = stage1_dir.parent / f"proper_eval{suffix}_{args.n_way}way_top{args.top_k}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CLIP
    model, processor = load_clip_model(device)
    
    # Precompute text features for all concepts
    print("\nComputing text features...")
    concept_names = [c["name"] for c in SEED_DV_CONCEPTS]
    text_features = get_clip_text_features(model, processor, concept_names, device)
    print(f"  Text features shape: {text_features.shape}")
    
    # Evaluate
    results = []
    n_way_correct = 0
    two_way_animal_correct = 0
    two_way_category_correct = 0
    total = 0
    
    # Per-concept stats
    concept_stats = defaultdict(lambda: {"total": 0, "n_way_correct": 0})
    
    max_samples = args.max_samples or len(samples)
    
    print(f"\nEvaluating {min(max_samples, len(samples))} samples...")
    
    for sample in tqdm(samples[:max_samples]):
        idx = sample['sample_idx']
        gt_concept_id = sample['concept_id']
        
        # Load frame
        if args.use_blurry:
            frame_path = stage1_dir / "blurry" / f"{idx:05d}.png"
        else:
            frame_path = stage1_dir / "refined" / f"{idx:05d}.png"
        
        if not frame_path.exists():
            continue
        
        image = np.array(Image.open(frame_path))
        
        # Get image features
        image_features = get_clip_image_features(model, processor, image, device)
        
        # N-way classification
        n_way_acc, pred_idx = n_way_classification(
            image_features,
            text_features,
            gt_concept_id,
            n_way=args.n_way,
            top_k=args.top_k,
            n_trials=args.n_trials,
        )
        
        n_way_correct += n_way_acc
        
        # 2-way classification
        two_way_animal = two_way_classification(gt_concept_id, pred_idx, mode="animal")
        two_way_category = two_way_classification(gt_concept_id, pred_idx, mode="category")
        
        if two_way_animal:
            two_way_animal_correct += 1
        if two_way_category:
            two_way_category_correct += 1
        
        total += 1
        
        # Per-concept stats
        concept_stats[gt_concept_id]["total"] += 1
        concept_stats[gt_concept_id]["n_way_correct"] += n_way_acc
        
        result = {
            'sample_idx': idx,
            'subject_id': sample['subject_id'],
            'gt_concept_id': gt_concept_id,
            'gt_concept_name': SEED_DV_CONCEPTS[gt_concept_id]["name"],
            'pred_concept_id': pred_idx,
            'pred_concept_name': SEED_DV_CONCEPTS[pred_idx]["name"],
            'n_way_correct': n_way_acc,
            'two_way_animal_correct': two_way_animal,
            'two_way_category_correct': two_way_category,
        }
        results.append(result)
    
    # Summary
    n_way_accuracy = n_way_correct / total
    two_way_animal_accuracy = two_way_animal_correct / total
    two_way_category_accuracy = two_way_category_correct / total
    
    # Random baselines
    random_n_way = args.top_k / args.n_way
    n_animals = sum(1 for c in SEED_DV_CONCEPTS if c["has_animal"])
    random_animal = (n_animals/40)**2 + ((40-n_animals)/40)**2  # P(both animal) + P(both non-animal)
    
    print("\n" + "="*60)
    print("PROPER CLASSIFICATION RESULTS")
    print("="*60)
    print(f"\nSettings:")
    print(f"  N-way: {args.n_way}")
    print(f"  Top-K: {args.top_k}")
    print(f"  Trials: {args.n_trials}")
    print(f"  Frame type: {'blurry' if args.use_blurry else 'refined'}")
    
    print(f"\nResults (n={total}):")
    print(f"  {args.n_way}-way Top-{args.top_k} Accuracy: {n_way_accuracy*100:.1f}% (random: {random_n_way*100:.1f}%)")
    print(f"  2-way Animal Accuracy: {two_way_animal_accuracy*100:.1f}% (random: {random_animal*100:.1f}%)")
    print(f"  2-way Category Accuracy: {two_way_category_accuracy*100:.1f}%")
    
    print("\n" + "="*60)
    print("COMPARISON WITH EEG2Video")
    print("="*60)
    print(f"{'Metric':<25} {'EEG2Video':<15} {'Ours':<15} {'Random':<10}")
    print("-"*65)
    print(f"{'40-way Top-1 Acc':<25} {'15.9%':<15} {n_way_accuracy*100:.1f}%{'':<9} {random_n_way*100:.1f}%")
    print(f"{'2-way Acc (animal)':<25} {'79.8%':<15} {two_way_animal_accuracy*100:.1f}%{'':<9} {random_animal*100:.1f}%")
    
    # Per-concept breakdown
    print("\n" + "="*60)
    print("PER-CONCEPT ACCURACY")
    print("="*60)
    print(f"{'Concept':<25} {'Samples':<10} {'Accuracy':<10}")
    print("-"*45)
    for concept_id in sorted(concept_stats.keys()):
        stats = concept_stats[concept_id]
        if stats["total"] > 0:
            acc = stats["n_way_correct"] / stats["total"]
            name = SEED_DV_CONCEPTS[concept_id]["name"][:24]
            print(f"{name:<25} {stats['total']:<10} {acc*100:.1f}%")
    
    # Save results
    summary = {
        'settings': {
            'stage1_dir': str(stage1_dir),
            'use_blurry': args.use_blurry,
            'n_way': args.n_way,
            'top_k': args.top_k,
            'n_trials': args.n_trials,
            'stage1_settings': settings,
        },
        'metrics': {
            'n_way_accuracy': float(n_way_accuracy),
            'two_way_animal_accuracy': float(two_way_animal_accuracy),
            'two_way_category_accuracy': float(two_way_category_accuracy),
            'random_n_way': float(random_n_way),
            'random_animal': float(random_animal),
        },
        'total_samples': total,
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Per-concept summary
    concept_summary = []
    for concept_id in range(40):
        stats = concept_stats[concept_id]
        concept_summary.append({
            'concept_id': concept_id,
            'name': SEED_DV_CONCEPTS[concept_id]["name"],
            'category': SEED_DV_CONCEPTS[concept_id]["category"],
            'has_animal': SEED_DV_CONCEPTS[concept_id]["has_animal"],
            'total': stats["total"],
            'n_way_accuracy': stats["n_way_correct"] / stats["total"] if stats["total"] > 0 else 0,
        })
    
    with open(output_dir / "per_concept.json", 'w') as f:
        json.dump(concept_summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
