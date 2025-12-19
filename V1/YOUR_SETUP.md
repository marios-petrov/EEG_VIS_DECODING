# SETUP GUIDE - Marios @ /home/mpetrov/EEG_Reconstruction

## 🎯 YOUR EXACT SETUP

**Location:** `/home/mpetrov/EEG_Reconstruction`  
**Dataset:** NATVIEW EEG+fMRI dataset (22 subjects)  
**Hardware:** 8x V100 32GB GPUs  
**Tasks:** DME (Despicable Me), TP (The Present), Inscapes

---

## 📥 STEP 1: Download Files

Download these files to `/home/mpetrov/EEG_Reconstruction/`:

### Core Python Scripts (8 files):
1. [preprocess_natview.py](computer:///mnt/user-data/outputs/preprocess_natview.py) ⭐ **NEW** - For NATVIEW .set files
2. [finetune_unet_v100.py](computer:///mnt/user-data/outputs/finetune_unet_v100.py) - UNet fine-tuning
3. [train_v100.py](computer:///mnt/user-data/outputs/train_v100.py) - Adapter training
4. [eval_qual.py](computer:///mnt/user-data/outputs/eval_qual.py) - Video generation
5. [eval_quant.py](computer:///mnt/user-data/outputs/eval_quant.py) - Metrics
6. [models.py](computer:///mnt/user-data/outputs/models.py) - Model architectures
7. [utils.py](computer:///mnt/user-data/outputs/utils.py) - Utilities

### Documentation:
8. [NATVIEW_COMMANDS.md](computer:///mnt/user-data/outputs/NATVIEW_COMMANDS.md) ⭐⭐⭐ **READ THIS!**

---

## 🚀 STEP 2: Install Dependencies

```bash
cd /home/mpetrov/EEG_Reconstruction

# Install MNE for reading EEGLAB .set files
pip install mne

# Verify everything
python -c "import torch; print(f'{torch.cuda.device_count()} GPUs')"
python -c "import mne; print('MNE installed')"
```

---

## ⚡ STEP 3: Run the Pipeline

### Quick Test (2-3 hours)
```bash
# Test with just 2 subjects
CUDA_VISIBLE_DEVICES=0 python preprocess_natview.py \
    --data-root /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/data \
    --stim-root /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/stim \
    --output-dir ./preprocessed_dme_test \
    --task dme \
    --subjects sub-01 sub-02 \
    --runs 01 \
    --generate-captions

torchrun --nproc_per_node=8 finetune_unet_v100.py \
    --task dme \
    --preprocessed-dir ./preprocessed_dme_test \
    --output-dir ./unet_test \
    --epochs 20 \
    --batch-size 16

torchrun --nproc_per_node=8 train_v100.py \
    --task dme \
    --preprocessed-dir ./preprocessed_dme_test \
    --output-dir ./adapter_test \
    --clip-embeddings-path ./preprocessed_dme_test/dme_clip_text_embeddings.npy \
    --epochs 20 \
    --batch-size 32 \
    --num-subjects 2

CUDA_VISIBLE_DEVICES=0 python eval_qual.py \
    --adapter-checkpoint ./adapter_test/best_model.pth \
    --unet-checkpoint ./unet_test/unet_ema_ep020.pth \
    --preprocessed-dir ./preprocessed_dme_test \
    --task dme \
    --output-dir ./results_test \
    --num-samples 5
```

### Full Pipeline - All 22 Subjects (~12 hours)

```bash
# 1. Preprocess all subjects (~3 hours)
CUDA_VISIBLE_DEVICES=0 python preprocess_natview.py \
    --data-root /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/data \
    --stim-root /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/stim \
    --output-dir ./preprocessed_dme \
    --task dme \
    --runs 01 02 \
    --context-prompt "a scene from Despicable Me animated movie" \
    --generate-captions

# 2. Fine-tune UNet (~8 hours)
torchrun --nproc_per_node=8 finetune_unet_v100.py \
    --task dme \
    --preprocessed-dir ./preprocessed_dme \
    --output-dir ./unet_checkpoints_dme \
    --epochs 200 \
    --batch-size 16 \
    --precision fp16 \
    --xformers \
    --num-workers 8

# 3. Train adapter (~3 hours)
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
    --num-workers 8

# 4. Evaluate (~30 min)
CUDA_VISIBLE_DEVICES=0 python eval_qual.py \
    --adapter-checkpoint ./adapter_checkpoints_dme/best_model.pth \
    --unet-checkpoint ./unet_checkpoints_dme/unet_ema_ep200.pth \
    --preprocessed-dir ./preprocessed_dme \
    --task dme \
    --output-dir ./results_qual_dme \
    --num-samples 50

CUDA_VISIBLE_DEVICES=0 python eval_quant.py \
    --adapter-checkpoint ./adapter_checkpoints_dme/best_model.pth \
    --unet-checkpoint ./unet_checkpoints_dme/unet_ema_ep200.pth \
    --preprocessed-dir ./preprocessed_dme \
    --task dme \
    --output-dir ./results_quant_dme \
    --num-samples 100
```

---

## 📊 What to Expect

### Preprocessing Output:
```
./preprocessed_dme/
├── dme_eeg.npy                      [~3500, 63, 125]
├── dme_vae_latents_hd.npy          [~3500, 6, 4, 96, 96]
├── dme_captions_hd.json
├── dme_clip_text_embeddings.npy    [~3500, 6, 1, 1024]
└── dme_metadata.json
```

### Training:
- UNet: ~3 min/epoch × 200 = ~10 hours
- Adapter: ~1 min/epoch × 100 = ~2 hours
- GPU utilization: Should see >90% on all 8 GPUs

### Results:
- Videos in `./results_qual_dme/`
- Metrics in `./results_quant_dme/quantitative_results.json`

---

## 🐛 Quick Checks

### Verify Dataset
```bash
# Should see all subjects
ls /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/data/ | grep sub

# Should see videos
ls /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/stim/
```

### Monitor Training
```bash
# In another terminal
watch -n 1 nvidia-smi

# Check logs
tail -f unet_checkpoints_dme/*.log  # If you redirect output
```

### Verify Preprocessing Worked
```bash
python -c "
import numpy as np
eeg = np.load('./preprocessed_dme/dme_eeg.npy')
print(f'EEG shape: {eeg.shape}')
print(f'Expected: [~3500, 63, 125]')
print(f'Looks good!' if eeg.shape[1] == 63 else 'Check preprocessing!')
"
```

---

## 💡 Tips

1. **Start with quick test** - Verify everything works with 2 subjects
2. **Run preprocessing overnight** - Takes 2-3 hours per task
3. **Monitor GPU usage** - Should be >90% during training
4. **Use screen/tmux** - For long-running jobs
5. **Check disk space** - Need ~30-50 GB free

### Screen Commands:
```bash
# Start new screen session
screen -S eeg2video

# Detach: Ctrl+A then D
# Reattach: screen -r eeg2video
# List: screen -ls
```

---

## 🎯 Your Complete Workflow

```bash
cd /home/mpetrov/EEG_Reconstruction

# Start screen session
screen -S preprocessing

# Run quick test first
CUDA_VISIBLE_DEVICES=0 python preprocess_natview.py \
    --data-root /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/data \
    --stim-root /local-scratch/marios-datasets/EEG_FMRI_NATVIEW/stim \
    --output-dir ./preprocessed_dme_test \
    --task dme \
    --subjects sub-01 sub-02 \
    --runs 01 \
    --generate-captions

# If test works, run full preprocessing
# Then detach (Ctrl+A, D) and let it run

# In another screen for training
screen -S training

torchrun --nproc_per_node=8 finetune_unet_v100.py \
    --task dme \
    --preprocessed-dir ./preprocessed_dme_test \
    --output-dir ./unet_test \
    --epochs 200 \
    --batch-size 16

# Monitor in another terminal
watch -n 1 nvidia-smi
```

---

## 📚 Full Documentation

For complete details, see:
- **[NATVIEW_COMMANDS.md](computer:///mnt/user-data/outputs/NATVIEW_COMMANDS.md)** - All commands and options
- **[V100_COMMANDS.md](computer:///mnt/user-data/outputs/V100_COMMANDS.md)** - V100-specific optimization details
- **[README.md](computer:///mnt/user-data/outputs/README.md)** - General documentation

---

## ✅ Checklist

Before you start:
- [ ] Downloaded all 8 Python files
- [ ] Installed MNE: `pip install mne`
- [ ] Verified dataset paths exist
- [ ] Verified 8 GPUs available
- [ ] Have ~50 GB free disk space
- [ ] Read NATVIEW_COMMANDS.md

Ready to train:
- [ ] Preprocessed data successfully
- [ ] Checked preprocessed output shapes
- [ ] All 8 GPUs visible to PyTorch
- [ ] Started screen/tmux session

---

## 🎉 You're All Set!

Your exact commands are ready. Just:
1. Download the 8 files above
2. Run the quick test (2-3 hours)
3. If it works, run the full pipeline (~12 hours)

**Questions?** Check NATVIEW_COMMANDS.md for detailed explanations and troubleshooting!

Good luck with your EEG2Video experiments! 🚀
