#!/usr/bin/env python3
"""
Check captions file to see what's wrong
"""

import json
import sys
from pathlib import Path
from collections import Counter

def check_captions(captions_path):
    print("="*70)
    print(f"ANALYZING CAPTIONS FILE")
    print("="*70)
    print(f"File: {captions_path}\n")
    
    if not Path(captions_path).exists():
        print(f"❌ ERROR: File not found!")
        return
    
    # Load the file
    with open(captions_path, 'r') as f:
        data = json.load(f)
    
    print(f"JSON type: {type(data)}")
    
    # Handle different JSON structures
    if isinstance(data, dict):
        print(f"Keys in dict: {list(data.keys())}")
        
        if 'captions' in data:
            captions = data['captions']
            print(f"\n✓ Found 'captions' key")
        else:
            print(f"\n❌ No 'captions' key found!")
            print(f"Available keys: {list(data.keys())}")
            return
    elif isinstance(data, list):
        captions = data
        print(f"Direct list of captions")
    else:
        print(f"❌ Unknown structure: {type(data)}")
        return
    
    # Analyze captions
    print(f"\n" + "="*70)
    print(f"CAPTION STATISTICS")
    print("="*70)
    print(f"Total captions: {len(captions):,}")
    
    # Check types
    caption_types = Counter([type(c).__name__ for c in captions[:100]])
    print(f"\nCaption types (first 100): {dict(caption_types)}")
    
    # Count unique
    if len(captions) < 100000:  # Only if manageable
        unique_captions = set(captions)
        print(f"Unique captions: {len(unique_captions):,}")
        
        if len(unique_captions) <= 20:
            print(f"\n⚠️  WARNING: Only {len(unique_captions)} unique caption(s)!")
            print(f"\nAll unique captions:")
            for i, cap in enumerate(sorted(unique_captions), 1):
                print(f"  {i}. \"{cap}\"")
    else:
        # Sample for large files
        sample_captions = captions[::1000]  # Every 1000th
        unique_sample = set(sample_captions)
        print(f"Unique in sample (every 1000th): {len(unique_sample):,}")
    
    # Show examples
    print(f"\n" + "="*70)
    print(f"CAPTION EXAMPLES")
    print("="*70)
    
    print(f"\nFirst 10 captions:")
    for i in range(min(10, len(captions))):
        print(f"  [{i}]: \"{captions[i]}\"")
    
    print(f"\nMiddle 5 captions (around index {len(captions)//2}):")
    mid = len(captions) // 2
    for i in range(mid, min(mid+5, len(captions))):
        print(f"  [{i}]: \"{captions[i]}\"")
    
    print(f"\nLast 5 captions:")
    for i in range(max(0, len(captions)-5), len(captions)):
        print(f"  [{i}]: \"{captions[i]}\"")
    
    # Random samples
    import random
    if len(captions) > 20:
        print(f"\n5 Random samples:")
        random_indices = random.sample(range(len(captions)), min(5, len(captions)))
        for idx in sorted(random_indices):
            print(f"  [{idx}]: \"{captions[idx]}\"")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        captions_path = sys.argv[1]
    else:
        # Default paths to check
        possible_paths = [
            "/local-scratch/marios-datasets/EEG_FMRI_NATVIEW/preprocessed_dme_sd35/dme_captions_hd.json",
            "/local-scratch/marios-datasets/EEG_FMRI_NATVIEW/preprocessed_dme/dme_captions_hd.json",
        ]
        
        captions_path = None
        for path in possible_paths:
            if Path(path).exists():
                captions_path = path
                break
        
        if captions_path is None:
            print("❌ No captions file found in default locations!")
            print("\nSearching for caption files...")
            import glob
            found = glob.glob("/local-scratch/marios-datasets/**/*caption*.json", recursive=True)
            if found:
                print(f"\nFound {len(found)} caption files:")
                for f in found:
                    print(f"  - {f}")
                print(f"\nUsing: {found[0]}")
                captions_path = found[0]
            else:
                print("No caption files found!")
                sys.exit(1)
    
    check_captions(captions_path)
