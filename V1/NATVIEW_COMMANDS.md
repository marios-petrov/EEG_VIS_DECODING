# NATVIEW Dataset - Exact Commands for Your Setup

## 📁 Your Paths
- **Working directory:** `/home/mpetrov/EEG_Reconstruction`
- **Dataset:** `/local-scratch/marios-datasets/EEG_FMRI_NATVIEW/data`
- **Videos:** `/local-scratch/marios-datasets/EEG_FMRI_NATVIEW/stim`
- **Subjects:** sub-01 through sub-22 (22 subjects total)
- **GPUs:** 8x V100 32GB

---

## 🚀 Complete Pipeline Commands

### Step 0: Setup

```bash
# Navigate to working directory
cd /home/mpetrov/EEG_Reconstruction

# Install MNE for reading .set files (if not already installed)
pip install mne

# Verify GPUs
nvidia-smi
python -c "import torch; print(f'{torch.cuda.device_count()} GPUs available')"
```

---

### Step 1: Preprocessing

#### Process All Subjects for Despicable Me (DME)

```bash
# DME task (both run-01 and run-02, all 22 subjects)
CUDA_VISIBLE_DEVICES=0 python preprocess_natview.py \
    --data-root /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/data \
    --stim-root /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/stim \
    --output-dir ./preprocessed_dme \
    --task dme \
    --runs 01 02 \
    --video-fps 6 \
    --frames-per-segment 6 \
    --resolution 768 \
    --context-prompt "a scene from Despicable Me animated movie" \
    --generate-captions
```

**Time estimate:** ~2-3 hours for all 22 subjects (both runs)

#### Process The Present (TP)

```bash
# TP task (both runs, all subjects)
CUDA_VISIBLE_DEVICES=0 python preprocess_natview.py \
    --data-root /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/data \
    --stim-root /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/stim \
    --output-dir ./preprocessed_tp \
    --task tp \
    --runs 01 02 \
    --video-fps 6 \
    --frames-per-segment 6 \
    --resolution 768 \
    --context-prompt "a scene from The Present animated short film" \
    --generate-captions
```

#### Process Inscapes

```bash
# Inscapes task (no runs, all subjects)
CUDA_VISIBLE_DEVICES=0 python preprocess_natview.py \
    --data-root /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/data \
    --stim-root /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/stim \
    --output-dir ./preprocessed_inscapes \
    --task inscapes \
    --video-fps 6 \
    --frames-per-segment 6 \
    --resolution 768 \
    --context-prompt "abstract moving shapes and patterns" \
    --generate-captions
```

#### Process Specific Subjects Only (for testing)

```bash
# Test with just 2 subjects first
CUDA_VISIBLE_DEVICES=0 python preprocess_natview.py \
    --data-root /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/data \
    --stim-root /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/stim \
    --output-dir ./preprocessed_dme_test \
    --task dme \
    --subjects sub-01 sub-02 \
    --runs 01 02 \
    --video-fps 6 \
    --frames-per-segment 6 \
    --resolution 768 \
    --generate-captions
```

---

### Step 2: UNet Fine-tuning (All 8 GPUs)

```bash
# Fine-tune UNet for DME task
torchrun --nproc_per_node=8 finetune_unet_v100.py \
    --task dme \
    --preprocessed-dir ./preprocessed_dme \
    --output-dir ./unet_checkpoints_dme \
    --epochs 200 \
    --batch-size 16 \
    --grad-accum 1 \
    --lr 1e-4 \
    --precision fp16 \
    --xformers \
    --num-workers 8
```

**Time estimate:** ~8 hours for 200 epochs

**For TP:**
```bash
torchrun --nproc_per_node=8 finetune_unet_v100.py \
    --task tp \
    --preprocessed-dir ./preprocessed_tp \
    --output-dir ./unet_checkpoints_tp \
    --epochs 200 \
    --batch-size 16 \
    --lr 1e-4 \
    --precision fp16 \
    --xformers \
    --num-workers 8
```

---

### Step 3: Adapter Training (All 8 GPUs)

```bash
# Train adapter for DME with 22 subjects
torchrun --nproc_per_node=8 train_v100.py \
    --task dme \
    --preprocessed-dir ./preprocessed_dme \
    --output-dir ./adapter_checkpoints_dme \
    --clip-embeddings-path ./preprocessed_dme/dme_clip_text_embeddings.npy \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --use-semantic-loss \
    --semantic-loss-weight 0.1 \
    --use-glmnet \
    --num-subjects 22 \
    --num-workers 8
```

**Time estimate:** ~2-3 hours for 100 epochs

**With Weights & Biases logging:**
```bash
torchrun --nproc_per_node=8 train_v100.py \
    --task dme \
    --preprocessed-dir ./preprocessed_dme \
    --output-dir ./adapter_checkpoints_dme \
    --clip-embeddings-path ./preprocessed_dme/dme_clip_text_embeddings.npy \
    --epochs 100 \
    --batch-size 32 \
    --use-semantic-loss \
    --use-glmnet \
    --num-subjects 22 \
    --wandb \
    --wandb-project natview-eeg2video \
    --num-workers 8
```

---

### Step 4: Evaluation

#### Qualitative (Generate Videos)

```bash
# Generate 50 test videos
CUDA_VISIBLE_DEVICES=0 python eval_qual.py \
    --adapter-checkpoint ./adapter_checkpoints_dme/best_model.pth \
    --unet-checkpoint ./unet_checkpoints_dme/unet_ema_ep200.pth \
    --preprocessed-dir ./preprocessed_dme \
    --task dme \
    --output-dir ./results_qual_dme \
    --num-samples 50 \
    --num-inference-steps 50 \
    --guidance-scale 7.5
```

**Time estimate:** ~15 minutes for 50 samples

#### Quantitative (Compute Metrics)

```bash
# Compute metrics on 100 samples
CUDA_VISIBLE_DEVICES=0 python eval_quant.py \
    --adapter-checkpoint ./adapter_checkpoints_dme/best_model.pth \
    --unet-checkpoint ./unet_checkpoints_dme/unet_ema_ep200.pth \
    --preprocessed-dir ./preprocessed_dme \
    --task dme \
    --output-dir ./results_quant_dme \
    --num-samples 100 \
    --num-inference-steps 50
```

**Time estimate:** ~30 minutes for 100 samples

---

## 🎯 Recommended Workflow

### Option A: Quick Test (2-3 hours)
```bash
# 1. Preprocess 2 subjects
CUDA_VISIBLE_DEVICES=0 python preprocess_natview.py \
    --data-root /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/data \
    --stim-root /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/stim \
    --output-dir ./preprocessed_dme_test \
    --task dme \
    --subjects sub-01 sub-02 \
    --runs 01 \
    --generate-captions

# 2. Quick UNet fine-tuning (20 epochs)
torchrun --nproc_per_node=8 finetune_unet_v100.py \
    --task dme \
    --preprocessed-dir ./preprocessed_dme_test \
    --output-dir ./unet_test \
    --epochs 20 \
    --batch-size 16

# 3. Quick adapter training (20 epochs)
torchrun --nproc_per_node=8 train_v100.py \
    --task dme \
    --preprocessed-dir ./preprocessed_dme_test \
    --output-dir ./adapter_test \
    --clip-embeddings-path ./preprocessed_dme_test/dme_clip_text_embeddings.npy \
    --epochs 20 \
    --batch-size 32 \
    --num-subjects 2

# 4. Generate a few videos
CUDA_VISIBLE_DEVICES=0 python eval_qual.py \
    --adapter-checkpoint ./adapter_test/best_model.pth \
    --unet-checkpoint ./unet_test/unet_ema_ep020.pth \
    --preprocessed-dir ./preprocessed_dme_test \
    --task dme \
    --output-dir ./results_test \
    --num-samples 5
```

### Option B: Full Pipeline (All Subjects) (~12-15 hours)
```bash
# Use the full commands from above
# 1. Preprocess all subjects (~3 hours)
# 2. Fine-tune UNet (~8 hours)
# 3. Train adapter (~3 hours)
# 4. Evaluate (~1 hour)
```

---

## 📊 Expected Data Sizes

### After Preprocessing (per task):
```
DME (22 subjects, 2 runs each):
  - EEG: [~3500, 63, 125] ≈ 100 MB
  - Video frames: [~3500, 6, 768, 768, 3] ≈ 150 GB
  - VAE latents: [~3500, 6, 4, 96, 96] ≈ 3 GB
  - CLIP embeddings: [~3500, 6, 1, 1024] ≈ 100 MB

Total per task: ~3-5 GB (without raw video frames)
```

### Disk Space Requirements:
- Preprocessed data: ~10-15 GB (all 3 tasks)
- UNet checkpoints: ~5 GB per task
- Adapter checkpoints: ~500 MB per task
- Results: ~2-5 GB per evaluation

**Total: ~30-50 GB recommended**

---

## 🔧 Parallel Processing Multiple Tasks

You can process multiple tasks simultaneously on different GPUs:

```bash
# Terminal 1: DME on GPUs 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_v100.py \
    --task dme \
    --preprocessed-dir ./preprocessed_dme \
    --output-dir ./adapter_dme \
    --clip-embeddings-path ./preprocessed_dme/dme_clip_text_embeddings.npy \
    --batch-size 32 \
    --num-subjects 22 &

# Terminal 2: TP on GPUs 4-7
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train_v100.py \
    --task tp \
    --preprocessed-dir ./preprocessed_tp \
    --output-dir ./adapter_tp \
    --clip-embeddings-path ./preprocessed_tp/tp_clip_text_embeddings.npy \
    --batch-size 32 \
    --num-subjects 22
```

---

## 🐛 Troubleshooting

### Check Your Dataset Structure
```bash
# Verify paths exist
ls /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/data/sub-01/ses-01/eeg/
ls /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/stim/

# Count subjects
ls -d /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/data/sub-* | wc -l  # Should be 22
```

### Monitor GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Check Preprocessed Output
```bash
python -c "
import numpy as np
eeg = np.load('./preprocessed_dme/dme_eeg.npy')
print(f'EEG shape: {eeg.shape}')
print(f'Segments: {eeg.shape[0]}')
print(f'Channels: {eeg.shape[1]}')
print(f'Time steps: {eeg.shape[2]}')
"
```

### If You Run Out of Memory During Preprocessing
```bash
# Reduce resolution
--resolution 512  # Instead of 768

# Or process subjects in batches
--subjects sub-01 sub-02 sub-03 sub-04 sub-05
# Then run again with next batch
--subjects sub-06 sub-07 sub-08 sub-09 sub-10
```

---

## 💡 Performance Tips

1. **Start with DME task** - It has the most engaging content
2. **Test with 2 subjects first** - Verify everything works
3. **Use all 8 GPUs** - Commands above are optimized for this
4. **Monitor GPU utilization** - Should be >90% during training
5. **Use wandb** - Track experiments across all tasks
6. **Preprocess overnight** - It takes 2-3 hours per task
7. **Train during the day** - You can monitor progress

---

## 📈 Expected Results

### DME (Despicable Me):
- Should achieve SSIM > 0.75
- PSNR > 27 dB
- Good reconstruction of character movements

### TP (The Present):
- Should achieve SSIM > 0.72
- PSNR > 26 dB
- Good reconstruction of animation

### Inscapes:
- Should achieve SSIM > 0.70
- PSNR > 25 dB
- Abstract patterns are harder to reconstruct

---

## 🎉 You're Ready!

Run the commands in order and you'll have a complete EEG-to-video reconstruction system trained on 22 subjects!

**Estimated total time:** 
- Quick test: 2-3 hours
- Full pipeline (one task): 12-15 hours
- All three tasks: 36-45 hours (can parallelize)
