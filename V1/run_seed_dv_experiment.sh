#!/bin/bash
# Run full SEED-DV experiment pipeline for comparison with EEG2Video (NeurIPS 2024)
#
# IMPORTANT: Run this script INSIDE the docker container, e.g.:
#   docker run --gpus all --rm -v /local-scratch:/local-scratch -v /home/mpetrov:/home/mpetrov \
#     --shm-size=32g eeg2video:latest bash /home/mpetrov/EEG_Reconstruction/run_seed_dv_experiment.sh
#
# Or run steps individually for long-running jobs (use tmux!):
#   Step 1: python preprocess_seed_dv.py ...
#   Step 2: torchrun train_eegmamba_seed_dv.py ...
#   Step 3: python eval_eegmamba_seed_dv.py ...

set -e

# Configuration
SEED_DIR="/local-scratch/SEED"
OUTPUT_BASE="/local-scratch/marios-datasets/SEED"
PREPROCESSED_DIR="${OUTPUT_BASE}/preprocessed_sd35"
CHECKPOINT_DIR="${OUTPUT_BASE}/checkpoints/eegmamba_sd35"
EVAL_DIR="${OUTPUT_BASE}/eval_results"

SCRIPTS_DIR="/home/mpetrov/EEG_Reconstruction"

echo "=============================================="
echo "SEED-DV Experiment Pipeline"
echo "Comparing EEGMamba + SD 3.5 vs EEG2Video"
echo "=============================================="
echo "SEED_DIR: ${SEED_DIR}"
echo "OUTPUT_BASE: ${OUTPUT_BASE}"
echo ""

# Check if SEED data exists
if [ ! -d "${SEED_DIR}/EEG" ]; then
    echo "ERROR: SEED dataset not found at ${SEED_DIR}"
    echo "Expected structure: ${SEED_DIR}/EEG/ and ${SEED_DIR}/Video/"
    exit 1
fi

# Step 1: Preprocess
if [ ! -f "${PREPROCESSED_DIR}/index.json" ]; then
    echo ""
    echo "Step 1: Preprocessing SEED-DV dataset..."
    echo "----------------------------------------"
    echo "This will take ~2-3 hours (VAE encoding for 20 subjects)"
    
    python ${SCRIPTS_DIR}/preprocess_seed_dv.py \
        --seed-dir ${SEED_DIR} \
        --output-dir ${PREPROCESSED_DIR} \
        --target-size 768 \
        --subjects 1-20
else
    echo ""
    echo "Step 1: Preprocessing already complete, skipping..."
    echo "  Found: ${PREPROCESSED_DIR}/index.json"
fi

# Step 2: Train
if [ ! -f "${CHECKPOINT_DIR}/adapter_best.pt" ]; then
    echo ""
    echo "Step 2: Training EEGMamba adapter..."
    echo "------------------------------------"
    echo "This will take ~6-8 hours on 8 GPUs"
    
    torchrun --nproc_per_node=8 ${SCRIPTS_DIR}/train_eegmamba_seed_dv.py \
        --preprocessed-dir ${PREPROCESSED_DIR} \
        --output-dir ${CHECKPOINT_DIR} \
        --epochs 100 \
        --batch-size 32 \
        --subjects 1-20
else
    echo ""
    echo "Step 2: Training already complete, skipping..."
    echo "  Found: ${CHECKPOINT_DIR}/adapter_best.pt"
fi

# Step 3: Evaluate (both with and without captions)
echo ""
echo "Step 3: Evaluating..."
echo "--------------------"

# Evaluate WITH captions
echo ""
echo "3a. Evaluating WITH captions..."
python ${SCRIPTS_DIR}/eval_eegmamba_seed_dv.py \
    --checkpoint ${CHECKPOINT_DIR}/adapter_best.pt \
    --preprocessed-dir ${PREPROCESSED_DIR} \
    --output-dir ${EVAL_DIR} \
    --subjects 1-20

# Evaluate WITHOUT captions (ablation)
echo ""
echo "3b. Evaluating WITHOUT captions (ablation)..."
python ${SCRIPTS_DIR}/eval_eegmamba_seed_dv.py \
    --checkpoint ${CHECKPOINT_DIR}/adapter_best.pt \
    --preprocessed-dir ${PREPROCESSED_DIR} \
    --output-dir ${EVAL_DIR} \
    --subjects 1-20 \
    --no-captions

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "Results saved to: ${EVAL_DIR}"
echo "=============================================="

# Print comparison for both modes
echo ""
echo "========================================="
echo "RESULTS WITH CAPTIONS:"
echo "========================================="
cat ${EVAL_DIR}/with_captions/summary.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
m = data['metrics']
print(f\"SSIM: {m['ssim_mean']:.3f} ± {m['ssim_std']:.2f}\")
print(f\"40-way acc: {m['acc_40way']*100:.1f}%\")
print(f\"2-way acc: {m['acc_2way']*100:.1f}%\")
"

echo ""
echo "========================================="
echo "RESULTS WITHOUT CAPTIONS (ablation):"
echo "========================================="
cat ${EVAL_DIR}/no_captions/summary.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
m = data['metrics']
print(f\"SSIM: {m['ssim_mean']:.3f} ± {m['ssim_std']:.2f}\")
print(f\"40-way acc: {m['acc_40way']*100:.1f}%\")
print(f\"2-way acc: {m['acc_2way']*100:.1f}%\")
"

echo ""
echo "========================================="
echo "COMPARISON WITH EEG2Video (NeurIPS 2024):"
echo "========================================="
python3 -c "
import json

with open('${EVAL_DIR}/with_captions/summary.json') as f:
    with_cap = json.load(f)['metrics']
with open('${EVAL_DIR}/no_captions/summary.json') as f:
    no_cap = json.load(f)['metrics']

print(f\"{'Metric':<15} {'EEG2Video':<12} {'Ours+Cap':<12} {'Ours-Cap':<12}\")
print('-'*51)
print(f\"{'SSIM':<15} {'0.256':<12} {with_cap['ssim_mean']:.3f}{'':<8} {no_cap['ssim_mean']:.3f}\")
print(f\"{'40-way acc':<15} {'15.9%':<12} {with_cap['acc_40way']*100:.1f}%{'':<7} {no_cap['acc_40way']*100:.1f}%\")
print(f\"{'2-way acc':<15} {'79.8%':<12} {with_cap['acc_2way']*100:.1f}%{'':<7} {no_cap['acc_2way']*100:.1f}%\")
"
