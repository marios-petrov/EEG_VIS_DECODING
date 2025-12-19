#!/usr/bin/env python3
"""
Pre-download Stable Diffusion 2.1 models to /local-scratch/hf-cache
Run this ONCE before training to avoid downloading during multi-GPU runs
"""

import os
import sys
from pathlib import Path

# CRITICAL: Set cache location BEFORE importing transformers/diffusers
cache_dir = Path("/local-scratch/hf-cache")
cache_dir.mkdir(parents=True, exist_ok=True)

os.environ['HF_HOME'] = str(cache_dir)
os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir / "hub")
os.environ['TRANSFORMERS_CACHE'] = str(cache_dir / "transformers")
os.environ['TORCH_HOME'] = str(cache_dir / "torch")

print("="*70)
print("STABLE DIFFUSION 2.1 MODEL DOWNLOAD")
print("="*70)
print(f"Cache directory: {cache_dir}")
print()

# Check disk space
print("📊 Disk space before download:")
os.system(f"df -h /local-scratch | tail -1")
print()

# Now import (after setting env vars!)
print("📦 Importing libraries...")
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

model_name = "stabilityai/stable-diffusion-2-1"

print(f"\n🎨 Downloading from: {model_name}")
print("="*70)

try:
    print("\n1️⃣  Downloading VAE (~335 MB)...")
    vae = AutoencoderKL.from_pretrained(
        model_name, 
        subfolder="vae",
        cache_dir=cache_dir
    )
    print("   ✓ VAE downloaded")
    del vae
    
    print("\n2️⃣  Downloading UNet (~3.5 GB, this will take a few minutes)...")
    unet = UNet2DConditionModel.from_pretrained(
        model_name, 
        subfolder="unet",
        cache_dir=cache_dir
    )
    print("   ✓ UNet downloaded")
    del unet
    
    print("\n3️⃣  Downloading Text Encoder (~1.7 GB)...")
    text_encoder = CLIPTextModel.from_pretrained(
        model_name, 
        subfolder="text_encoder",
        cache_dir=cache_dir
    )
    print("   ✓ Text encoder downloaded")
    del text_encoder
    
    print("\n4️⃣  Downloading Tokenizer (~2 MB)...")
    tokenizer = CLIPTokenizer.from_pretrained(
        model_name, 
        subfolder="tokenizer",
        cache_dir=cache_dir
    )
    print("   ✓ Tokenizer downloaded")
    del tokenizer
    
    print("\n5️⃣  Downloading Scheduler (~500 KB)...")
    scheduler = DDPMScheduler.from_pretrained(
        model_name, 
        subfolder="scheduler",
        cache_dir=cache_dir
    )
    print("   ✓ Scheduler downloaded")
    del scheduler
    
    print("\n" + "="*70)
    print("✅ ALL MODELS SUCCESSFULLY DOWNLOADED")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ ERROR during download: {e}")
    sys.exit(1)

# Show cache size
print(f"\n📊 Cache directory size:")
os.system(f"du -sh {cache_dir}")

print(f"\n📊 Disk space after download:")
os.system(f"df -h /local-scratch | tail -1")

print("\n" + "="*70)
print("💡 NEXT STEPS")
print("="*70)
print("Models are cached at:")
print(f"   {cache_dir}")
print("\nMake sure these environment variables are set before training:")
print(f"   export HF_HOME={cache_dir}")
print(f"   export HUGGINGFACE_HUB_CACHE={cache_dir}/hub")
print(f"   export TRANSFORMERS_CACHE={cache_dir}/transformers")
print("\nRun training with:")
print("   torchrun --nproc_per_node=8 finetune_unet_v100_memeff.py ...")
print("="*70)
