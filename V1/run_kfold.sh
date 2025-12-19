#!/bin/bash
# Run all 5 folds of k-fold cross-validation
# Usage: ./run_kfold.sh

set -e

# Configuration
PREPROCESSED_DIR="/local-scratch/marios-datasets/EEG_FMRI_NATVIEW/preprocessed_dme_sd35"
BASE_OUTPUT_DIR="/local-scratch/marios-datasets/checkpoints/eegmamba_kfold"
EVAL_OUTPUT_DIR="/local-scratch/marios-datasets/eval_results/kfold"
TASK="dme"
EPOCHS=100
BATCH_SIZE=32
NUM_GPUS=8

echo "=============================================="
echo "K-FOLD CROSS-VALIDATION (5 FOLDS)"
echo "=============================================="
echo "Preprocessed dir: $PREPROCESSED_DIR"
echo "Output dir: $BASE_OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "GPUs: $NUM_GPUS"
echo ""

# Train all folds
for FOLD in 0 1 2 3 4; do
    echo ""
    echo "=============================================="
    echo "TRAINING FOLD $FOLD"
    echo "=============================================="
    
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/fold${FOLD}"
    
    docker run --gpus all --rm \
        -v /local-scratch:/local-scratch \
        -v /home/mpetrov:/home/mpetrov \
        --shm-size=32g \
        eeg2video:latest \
        torchrun --nproc_per_node=$NUM_GPUS \
            /home/mpetrov/EEG_Reconstruction/train_eegmamba_kfold.py \
            --fold $FOLD \
            --task $TASK \
            --preprocessed-dir $PREPROCESSED_DIR \
            --output-dir $OUTPUT_DIR \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE
    
    echo "✓ Fold $FOLD training complete"
done

echo ""
echo "=============================================="
echo "ALL TRAINING COMPLETE - STARTING EVALUATION"
echo "=============================================="

# Evaluate all folds
for FOLD in 0 1 2 3 4; do
    echo ""
    echo "=============================================="
    echo "EVALUATING FOLD $FOLD"
    echo "=============================================="
    
    CHECKPOINT="${BASE_OUTPUT_DIR}/fold${FOLD}/adapter_best.pt"
    EVAL_DIR="${EVAL_OUTPUT_DIR}/fold${FOLD}"
    
    docker run --gpus all --rm \
        -v /local-scratch:/local-scratch \
        -v /home/mpetrov:/home/mpetrov \
        --shm-size=16g \
        eeg2video:latest \
        python /home/mpetrov/EEG_Reconstruction/eval_eegmamba_kfold.py \
            --checkpoint $CHECKPOINT \
            --preprocessed-dir $PREPROCESSED_DIR \
            --output-dir $EVAL_DIR \
            --task $TASK \
            --skip-video
    
    echo "✓ Fold $FOLD evaluation complete"
done

echo ""
echo "=============================================="
echo "AGGREGATING RESULTS"
echo "=============================================="

docker run --gpus all --rm \
    -v /local-scratch:/local-scratch \
    -v /home/mpetrov:/home/mpetrov \
    eeg2video:latest \
    python /home/mpetrov/EEG_Reconstruction/eval_eegmamba_kfold.py \
        --aggregate-results $EVAL_OUTPUT_DIR

echo ""
echo "=============================================="
echo "K-FOLD CROSS-VALIDATION COMPLETE!"
echo "=============================================="
echo "Results: ${EVAL_OUTPUT_DIR}/aggregated_results.json"
