#!/usr/bin/env python3
"""
SEED-DV Classification Evaluation (40-way, 2-way Accuracy)

Evaluates semantic-level reconstruction quality using CLIP classifier.

Metrics (matching EEG2Video paper):
- 40-way Top-1 Accuracy: Paper reports 13.8% (frame), 15.9% (video)
- 2-way Accuracy (Animal/Non-animal): Paper reports 77.4% (frame), 79.8% (video)

SEED-DV Concept Mapping (1-indexed):
  1-11:  Animals (cat, dog, elephant, horse, panda, rabbit, bird, fish, jellyfish, shark, turtle)
  12-40: Non-animals (flower...ship)

Usage:
    python eval_seed_dv_classification.py \
        --generated-dir /path/to/stage1_results \
        --output-dir /path/to/eval_output
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
from typing import Dict, List, Tuple
import random
from collections import defaultdict

# ============================================================================
# SEED-DV CONCEPT DEFINITIONS (1-indexed as in paper)
# ============================================================================
SEED_DV_CONCEPTS_1INDEXED = {
    # Land Animal (1-7)
    1: {"name": "cat", "coarse": "land_animal", "is_animal": True},
    2: {"name": "dog", "coarse": "land_animal", "is_animal": True},
    3: {"name": "elephant", "coarse": "land_animal", "is_animal": True},
    4: {"name": "horse", "coarse": "land_animal", "is_animal": True},
    5: {"name": "panda", "coarse": "land_animal", "is_animal": True},
    6: {"name": "rabbit", "coarse": "land_animal", "is_animal": True},
    7: {"name": "bird", "coarse": "land_animal", "is_animal": True},
    # Water Animal (8-11)
    8: {"name": "fish", "coarse": "water_animal", "is_animal": True},
    9: {"name": "jellyfish", "coarse": "water_animal", "is_animal": True},
    10: {"name": "shark", "coarse": "water_animal", "is_animal": True},
    11: {"name": "turtle", "coarse": "water_animal", "is_animal": True},
    # Plant (12-14)
    12: {"name": "flower", "coarse": "plant", "is_animal": False},
    13: {"name": "mushroom", "coarse": "plant", "is_animal": False},
    14: {"name": "tree", "coarse": "plant", "is_animal": False},
    # Exercise (15-18)
    15: {"name": "boxing", "coarse": "exercise", "is_animal": False},
    16: {"name": "dancing", "coarse": "exercise", "is_animal": False},
    17: {"name": "running", "coarse": "exercise", "is_animal": False},
    18: {"name": "skiing", "coarse": "exercise", "is_animal": False},
    # Human (19-21)
    19: {"name": "couple", "coarse": "human", "is_animal": False},
    20: {"name": "face", "coarse": "human", "is_animal": False},
    21: {"name": "crowd", "coarse": "human", "is_animal": False},
    # Natural Scene (22-27)
    22: {"name": "beach", "coarse": "natural_scene", "is_animal": False},
    23: {"name": "buildings", "coarse": "natural_scene", "is_animal": False},
    24: {"name": "mountain", "coarse": "natural_scene", "is_animal": False},
    25: {"name": "road", "coarse": "natural_scene", "is_animal": False},
    26: {"name": "water", "coarse": "natural_scene", "is_animal": False},
    27: {"name": "fireworks", "coarse": "natural_scene", "is_animal": False},
    # Food (28-32)
    28: {"name": "banana", "coarse": "food", "is_animal": False},
    29: {"name": "cake", "coarse": "food", "is_animal": False},
    30: {"name": "drink", "coarse": "food", "is_animal": False},
    31: {"name": "pizza", "coarse": "food", "is_animal": False},
    32: {"name": "watermelon", "coarse": "food", "is_animal": False},
    # Musical (33-35)
    33: {"name": "drum", "coarse": "musical", "is_animal": False},
    34: {"name": "guitar", "coarse": "musical", "is_animal": False},
    35: {"name": "piano", "coarse": "musical", "is_animal": False},
    # Transportation (36-40)
    36: {"name": "bike", "coarse": "transportation", "is_animal": False},
    37: {"name": "car", "coarse": "transportation", "is_animal": False},
    38: {"name": "hot air balloon", "coarse": "transportation", "is_animal": False},
    39: {"name": "airplane", "coarse": "transportation", "is_animal": False},
    40: {"name": "ship", "coarse": "transportation", "is_animal": False},
}

# Convert to 0-indexed list for CLIP classification
SEED_DV_CONCEPTS = [SEED_DV_CONCEPTS_1INDEXED[i] for i in range(1, 41)]

N_ANIMAL = 11  # Concepts 1-11
N_TOTAL = 40


def get_concept_info(concept_id: int, is_1indexed: bool = True) -> Dict:
    """Get concept info from ID"""
    if is_1indexed:
        return SEED_DV_CONCEPTS_1INDEXED.get(concept_id, {"name": "unknown", "is_animal": False})
    else:
        if 0 <= concept_id < 40:
            return SEED_DV_CONCEPTS[concept_id]
        return {"name": "unknown", "is_animal": False}


def load_clip_model(device):
    """Load CLIP ViT-L/14 (same as paper)"""
    from transformers import CLIPProcessor, CLIPModel
    
    print("Loading CLIP ViT-L/14...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        torch_dtype=torch.float16,
    ).to(device).eval()
    print("  ✓ Loaded")
    
    return model, processor


def get_text_features(model, processor, device) -> torch.Tensor:
    """Compute CLIP text features for all 40 concepts"""
    # Multiple prompt templates (standard CLIP evaluation)
    templates = [
        "a photo of a {}",
        "a video of a {}",
        "a video showing a {}",
        "a photo of the {}",
    ]
    
    all_features = []
    
    for concept in SEED_DV_CONCEPTS:
        name = concept["name"]
        
        template_features = []
        for template in templates:
            text = template.format(name)
            inputs = processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                features = model.get_text_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
            
            template_features.append(features)
        
        avg_features = torch.stack(template_features).mean(dim=0)
        avg_features = avg_features / avg_features.norm(dim=-1, keepdim=True)
        all_features.append(avg_features)
    
    return torch.cat(all_features, dim=0)  # [40, D]


def classify_image(
    image: np.ndarray,
    model,
    processor,
    text_features: torch.Tensor,
    device,
) -> Tuple[np.ndarray, int]:
    """Classify image using CLIP"""
    pil_image = Image.fromarray(image)
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    similarities = (image_features @ text_features.T)[0]
    probs = F.softmax(similarities * 100, dim=0).cpu().numpy()
    
    return probs, int(similarities.argmax().item())


def n_way_top_k(
    probs: np.ndarray,
    gt_idx: int,  # 0-indexed
    n_way: int = 40,
    top_k: int = 1,
    n_trials: int = 10,
) -> float:
    """
    N-way top-K accuracy as per paper.
    
    For n_way < 40: Sample N-1 distractors + GT, check if GT in top-K.
    Average over multiple trials for stable results.
    """
    if n_way >= len(probs):
        # Full 40-way
        top_indices = np.argsort(probs)[-top_k:]
        return float(gt_idx in top_indices)
    
    # Sample-based N-way
    successes = 0
    for _ in range(n_trials):
        others = [i for i in range(len(probs)) if i != gt_idx]
        distractors = random.sample(others, n_way - 1)
        selected = distractors + [gt_idx]
        
        selected_probs = probs[selected]
        gt_pos = selected.index(gt_idx)
        
        top_positions = np.argsort(selected_probs)[-top_k:]
        if gt_pos in top_positions:
            successes += 1
    
    return successes / n_trials


def two_way_accuracy(gt_idx: int, pred_idx: int) -> bool:
    """2-way: Animal vs Non-animal (0-indexed)"""
    gt_animal = SEED_DV_CONCEPTS[gt_idx]["is_animal"]
    pred_animal = SEED_DV_CONCEPTS[pred_idx]["is_animal"]
    return gt_animal == pred_animal


def detect_indexing(samples: List[Dict]) -> bool:
    """Detect if concept_id is 0-indexed or 1-indexed"""
    concept_ids = [s['concept_id'] for s in samples[:100]]
    min_id = min(concept_ids)
    max_id = max(concept_ids)
    
    print(f"\nDetecting concept_id indexing:")
    print(f"  Min: {min_id}, Max: {max_id}")
    
    if min_id == 0 and max_id <= 39:
        print("  → 0-indexed (0-39)")
        return False
    elif min_id >= 1 and max_id <= 40:
        print("  → 1-indexed (1-40)")
        return True
    else:
        print(f"  → Unknown, assuming 0-indexed")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated-dir', type=str, required=True,
                       help='Directory with generated images')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use-blurry', action='store_true',
                       help='Evaluate blurry frames instead of refined')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--force-1indexed', action='store_true',
                       help='Force 1-indexed concept IDs')
    parser.add_argument('--n-way', type=int, default=40,
                       help='N for N-way classification (default: 40)')
    parser.add_argument('--top-k', type=int, default=1,
                       help='K for top-K accuracy (default: 1)')
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
    
    print(f"\n{'='*60}")
    print("SEED-DV CLASSIFICATION EVALUATION")
    print(f"{'='*60}")
    print(f"Generated dir: {gen_dir}")
    print(f"Samples: {len(samples)}")
    print(f"Frame type: {'blurry' if args.use_blurry else 'refined'}")
    print(f"N-way: {args.n_way}, Top-K: {args.top_k}")
    
    # Detect indexing
    is_1indexed = args.force_1indexed or detect_indexing(samples)
    
    # Show sample mapping
    print(f"\nSample concept mapping (first 5):")
    for s in samples[:5]:
        cid = s['concept_id']
        idx_0 = cid - 1 if is_1indexed else cid
        if 0 <= idx_0 < 40:
            name = SEED_DV_CONCEPTS[idx_0]["name"]
            print(f"  concept_id={cid} → '{name}'")
    
    # Output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = "_blurry" if args.use_blurry else "_refined"
        output_dir = gen_dir.parent / f"eval_classification{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CLIP
    model, processor = load_clip_model(device)
    text_features = get_text_features(model, processor, device)
    print(f"\nText features: {text_features.shape}")
    
    # Evaluate
    results = []
    acc_nway = 0
    acc_2way = 0
    total = 0
    skipped = 0
    
    concept_stats = defaultdict(lambda: {"total": 0, "correct_nway": 0, "correct_2way": 0})
    coarse_stats = defaultdict(lambda: {"total": 0, "correct_nway": 0, "correct_2way": 0})
    
    max_samples = args.max_samples or len(samples)
    
    print(f"\nEvaluating {min(max_samples, len(samples))} samples...")
    
    for sample in tqdm(samples[:max_samples]):
        idx = sample['sample_idx']
        cid_raw = sample['concept_id']
        
        # Convert to 0-indexed
        if is_1indexed:
            gt_idx = cid_raw - 1
        else:
            gt_idx = cid_raw
        
        if gt_idx < 0 or gt_idx >= 40:
            skipped += 1
            continue
        
        # Load frame
        if args.use_blurry:
            frame_path = gen_dir / "blurry" / f"{idx:05d}.png"
        else:
            frame_path = gen_dir / "refined" / f"{idx:05d}.png"
        
        if not frame_path.exists():
            skipped += 1
            continue
        
        image = np.array(Image.open(frame_path))
        probs, pred_idx = classify_image(image, model, processor, text_features, device)
        
        # Metrics (0-indexed)
        nway_acc = n_way_top_k(probs, gt_idx, n_way=args.n_way, top_k=args.top_k)
        is_2way = two_way_accuracy(gt_idx, pred_idx)
        
        acc_nway += nway_acc
        acc_2way += int(is_2way)
        total += 1
        
        # Stats
        concept_stats[gt_idx]["total"] += 1
        concept_stats[gt_idx]["correct_nway"] += nway_acc
        concept_stats[gt_idx]["correct_2way"] += int(is_2way)
        
        coarse = SEED_DV_CONCEPTS[gt_idx]["coarse"]
        coarse_stats[coarse]["total"] += 1
        coarse_stats[coarse]["correct_nway"] += nway_acc
        coarse_stats[coarse]["correct_2way"] += int(is_2way)
        
        results.append({
            'sample_idx': idx,
            'gt_concept_id': cid_raw,
            'gt_concept_idx': gt_idx,
            'gt_concept_name': SEED_DV_CONCEPTS[gt_idx]["name"],
            'gt_coarse': coarse,
            'gt_is_animal': SEED_DV_CONCEPTS[gt_idx]["is_animal"],
            'pred_concept_idx': pred_idx,
            'pred_concept_name': SEED_DV_CONCEPTS[pred_idx]["name"],
            'pred_is_animal': SEED_DV_CONCEPTS[pred_idx]["is_animal"],
            f'{args.n_way}way_top{args.top_k}_correct': nway_acc,
            '2way_correct': is_2way,
        })
    
    if total == 0:
        print("ERROR: No samples evaluated!")
        return
    
    # Final metrics
    final_nway = acc_nway / total
    final_2way = acc_2way / total
    
    # Random baselines
    random_nway = args.top_k / args.n_way
    p_animal = N_ANIMAL / N_TOTAL
    random_2way = p_animal**2 + (1 - p_animal)**2
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nTotal: {total} (skipped: {skipped})")
    print(f"\n{args.n_way}-way Top-{args.top_k} Accuracy: {final_nway*100:.1f}%  (random: {random_nway*100:.1f}%)")
    print(f"2-way Accuracy:              {final_2way*100:.1f}%  (random: {random_2way*100:.1f}%)")
    
    print("\n" + "="*60)
    print("COMPARISON WITH EEG2Video PAPER (Table 2)")
    print("="*60)
    print(f"{'Metric':<25} {'Paper':<12} {'Ours':<12} {'Δ':<8}")
    print("-"*55)
    
    delta_nway = final_nway*100 - 13.8
    delta_2way = final_2way*100 - 77.4
    
    print(f"{'40-way (frame)':<25} {'13.8%':<12} {final_nway*100:.1f}%{'':<6} {delta_nway:+.1f}%")
    print(f"{'2-way (frame)':<25} {'77.4%':<12} {final_2way*100:.1f}%{'':<6} {delta_2way:+.1f}%")
    print(f"{'40-way (video)':<25} {'15.9%':<12} {'N/A':<12}")
    print(f"{'2-way (video)':<25} {'79.8%':<12} {'N/A':<12}")
    
    # Per-coarse-class breakdown
    print("\n" + "="*60)
    print("PER-COARSE-CLASS ACCURACY")
    print("="*60)
    print(f"{'Coarse Class':<18} {f'{args.n_way}-way':<12} {'2-way':<12} {'N'}")
    print("-"*50)
    
    for coarse in ["land_animal", "water_animal", "plant", "exercise", "human",
                   "natural_scene", "food", "musical", "transportation"]:
        stats = coarse_stats[coarse]
        if stats["total"] > 0:
            nway = stats["correct_nway"] / stats["total"]
            tway = stats["correct_2way"] / stats["total"]
            print(f"{coarse:<18} {nway*100:.1f}%{'':<6} {tway*100:.1f}%{'':<6} {stats['total']}")
    
    # Top/Bottom concepts
    print("\n" + "="*60)
    print("TOP 5 BEST CONCEPTS")
    print("="*60)
    
    concept_accs = []
    for idx, stats in concept_stats.items():
        if stats["total"] > 0:
            acc = stats["correct_nway"] / stats["total"]
            concept_accs.append((idx, acc, stats["total"]))
    
    concept_accs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Concept':<20} {'Coarse':<15} {'Acc':<10} {'N'}")
    print("-"*55)
    for idx, acc, n in concept_accs[:5]:
        name = SEED_DV_CONCEPTS[idx]["name"]
        coarse = SEED_DV_CONCEPTS[idx]["coarse"]
        print(f"{name:<20} {coarse:<15} {acc*100:.1f}%{'':<5} {n}")
    
    print("\n" + "="*60)
    print("BOTTOM 5 WORST CONCEPTS")
    print("="*60)
    print(f"{'Concept':<20} {'Coarse':<15} {'Acc':<10} {'N'}")
    print("-"*55)
    for idx, acc, n in concept_accs[-5:]:
        name = SEED_DV_CONCEPTS[idx]["name"]
        coarse = SEED_DV_CONCEPTS[idx]["coarse"]
        print(f"{name:<20} {coarse:<15} {acc*100:.1f}%{'':<5} {n}")
    
    # Save
    summary = {
        'metrics': {
            f'{args.n_way}way_top{args.top_k}_accuracy': float(final_nway),
            '2way_accuracy': float(final_2way),
            f'random_{args.n_way}way': float(random_nway),
            'random_2way': float(random_2way),
        },
        'paper_comparison': {
            'paper_40way_frame': 0.138,
            'paper_2way_frame': 0.774,
            'paper_40way_video': 0.159,
            'paper_2way_video': 0.798,
        },
        'settings': {
            'generated_dir': str(gen_dir),
            'use_blurry': args.use_blurry,
            'is_1indexed': is_1indexed,
            'n_way': args.n_way,
            'top_k': args.top_k,
            'total_evaluated': total,
            'skipped': skipped,
            'n_animal_concepts': N_ANIMAL,
        },
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Per-concept breakdown
    concept_breakdown = []
    for idx in range(40):
        stats = concept_stats.get(idx, {"total": 0, "correct_nway": 0, "correct_2way": 0})
        if stats["total"] > 0:
            concept_breakdown.append({
                'concept_idx': idx,
                'concept_id_1indexed': idx + 1,
                'name': SEED_DV_CONCEPTS[idx]["name"],
                'coarse': SEED_DV_CONCEPTS[idx]["coarse"],
                'is_animal': SEED_DV_CONCEPTS[idx]["is_animal"],
                f'{args.n_way}way_accuracy': stats["correct_nway"] / stats["total"],
                '2way_accuracy': stats["correct_2way"] / stats["total"],
                'n_samples': stats["total"],
            })
    
    with open(output_dir / "per_concept.json", 'w') as f:
        json.dump(concept_breakdown, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
