#!/bin/bash
# Setup and Training Script for EEGMamba + SD 3.5 Adapter
# ========================================================

echo "=================================================="
echo "EEGMamba + SD 3.5 Video Reconstruction Setup"
echo "=================================================="

# Step 1: Clone EEGMamba repository
echo ""
echo "Step 1: Setting up EEGMamba..."
if [ ! -d "EEGMamba" ]; then
    git clone https://github.com/wjq-learning/EEGMamba.git
    echo "✓ EEGMamba cloned"
else
    echo "✓ EEGMamba already exists"
fi

# Step 2: Download pretrained weights
echo ""
echo "Step 2: Downloading pretrained EEGMamba weights..."
if [ ! -f "pretrained_EEGMamba.pth" ]; then
    wget https://huggingface.co/weighting666/EEGMamba/resolve/main/pretrained_EEGMamba.pth
    echo "✓ Pretrained weights downloaded"
else
    echo "✓ Pretrained weights already exist"
fi

# Step 3: Install dependencies
echo ""
echo "Step 3: Installing dependencies..."
pip install mamba-ssm einops diffusers>=0.30.0 transformers accelerate

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "Next Steps:"
echo ""
echo "1. Convert your latents to SD 3.5 (16-channel):"
echo "   python convert_latents_sd21_to_sd35.py \\"
echo "       --input-dir /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/preprocessed_dme/temp_subjects \\"
echo "       --output-dir /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/preprocessed_dme_sd35 \\"
echo "       --task dme"
echo ""
echo "2. Train the adapter (single GPU):"
echo "   python train_eegmamba_adapter.py \\"
echo "       --task dme \\"
echo "       --preprocessed-dir /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/preprocessed_dme_sd35 \\"
echo "       --output-dir /local-scratch/marios-datasets/checkpoints/eegmamba_adapter \\"
echo "       --eegmamba-pretrained pretrained_EEGMamba.pth \\"
echo "       --epochs 100 \\"
echo "       --batch-size 32 \\"
echo "       --lr 1e-4"
echo ""
echo "3. Train the adapter (multi-GPU):"
echo "   torchrun --nproc_per_node=8 train_eegmamba_adapter.py \\"
echo "       --task dme \\"
echo "       --preprocessed-dir /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/preprocessed_dme_sd35 \\"
echo "       --output-dir /local-scratch/marios-datasets/checkpoints/eegmamba_adapter \\"
echo "       --eegmamba-pretrained pretrained_EEGMamba.pth \\"
echo "       --epochs 100 \\"
echo "       --batch-size 32 \\"
echo "       --lr 1e-4"
echo ""
echo "=================================================="
