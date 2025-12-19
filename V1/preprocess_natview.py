#!/usr/bin/env python3
"""
Preprocessing pipeline for EEG2Video with NATVIEW dataset
Handles EEGLAB .set files and BIDS format
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPTextModel,
    CLIPTokenizer,
)

from utils import (
    read_video_at_fps,
    segment_frames,
    square_resize,
    seed_everything,
)
from models import load_diffusion_models


def load_eeglab_set(set_path: Path) -> Tuple[np.ndarray, float, List[str]]:
    """
    Load EEGLAB .set file
    Returns:
        eeg_data: [time_steps, channels] EEG data
        sfreq: Sampling frequency
        channel_names: List of channel names
    """
    try:
        import mne
    except ImportError:
        raise ImportError("MNE is required for reading .set files. Install with: pip install mne")
    
    # Load with MNE
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
    
    # Get data and sampling frequency
    eeg_data = raw.get_data().T  # [time_steps, channels]
    sfreq = raw.info['sfreq']
    channel_names = raw.ch_names
    
    print(f"  Loaded EEG: {eeg_data.shape} at {sfreq} Hz, channels: {len(channel_names)}")
    
    return eeg_data, sfreq, channel_names


def find_common_channels(
    data_root: Path,
    subjects: List[str],
    task: str,
    runs: List[Optional[str]],
) -> List[str]:
    """
    Find channels that are common across all subjects and runs
    """
    import mne
    
    all_channel_sets = []
    
    print("🔍 Scanning all subjects to find common channels...")
    for subject in subjects:
        for run in runs:
            subject_dir = data_root / subject / 'ses-01' / 'eeg'
            if run:
                base_name = f"{subject}_ses-01_task-{task}_run-{run}"
            else:
                base_name = f"{subject}_ses-01_task-{task}"
            
            eeg_path = subject_dir / f"{base_name}_eeg.set"
            
            if eeg_path.exists():
                try:
                    raw = mne.io.read_raw_eeglab(str(eeg_path), preload=False, verbose=False)
                    all_channel_sets.append(set(raw.ch_names))
                    print(f"  {subject} run-{run}: {len(raw.ch_names)} channels")
                except Exception as e:
                    print(f"  ⚠️  Could not read {eeg_path}: {e}")
    
    # Find intersection of all channel sets
    if not all_channel_sets:
        raise ValueError("No valid EEG files found!")
    
    common_channels = set.intersection(*all_channel_sets)
    common_channels_sorted = sorted(list(common_channels))
    
    print(f"\n✓ Found {len(common_channels_sorted)} common channels across all subjects")
    print(f"  Common channels: {', '.join(common_channels_sorted[:10])}..." if len(common_channels_sorted) > 10 else f"  Common channels: {', '.join(common_channels_sorted)}")
    
    return common_channels_sorted


def standardize_eeg_channels(
    eeg_data: np.ndarray,
    channel_names: List[str],
    target_channels: List[str],
) -> np.ndarray:
    """
    Standardize EEG data to use only target channels in specified order
    
    Args:
        eeg_data: [time_steps, channels] array
        channel_names: List of channel names in eeg_data
        target_channels: List of target channel names to keep
    
    Returns:
        standardized_eeg: [time_steps, len(target_channels)] array
    """
    # Create mapping
    channel_indices = []
    for target_ch in target_channels:
        if target_ch in channel_names:
            idx = channel_names.index(target_ch)
            channel_indices.append(idx)
        else:
            raise ValueError(f"Target channel {target_ch} not found in data!")
    
    # Select and reorder channels
    standardized_eeg = eeg_data[:, channel_indices]
    
    return standardized_eeg


def load_events(events_path: Path) -> List[Tuple[float, float, str]]:
    """
    Load events from events.tsv
    Returns:
        List of (onset, duration, trial_type) tuples
    """
    events = []
    with open(events_path, 'r') as f:
        header = f.readline().strip().split('\t')
        onset_idx = header.index('onset')
        duration_idx = header.index('duration')
        trial_idx = header.index('trial_type') if 'trial_type' in header else None
        
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) <= max(onset_idx, duration_idx):
                continue
            
            try:
                onset = float(parts[onset_idx])
                duration = float(parts[duration_idx])
                trial_type = parts[trial_idx] if trial_idx is not None and len(parts) > trial_idx else "segment"
                
                # Accept ALL events with valid onset and duration
                # Skip only if duration is 0 or negative
                if duration > 0:
                    events.append((onset, duration, trial_type))
            except (ValueError, IndexError):
                continue
    
    return events


def synchronize_eeg_video(
    eeg: np.ndarray,
    events: List[Tuple[float, float, str]],
    eeg_sample_rate: int,
    video_fps: int,
    frames_per_segment: int,
    eeg_time_steps: int = 125,
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Synchronize EEG with video events
    
    Returns:
        eeg_segments: List of [eeg_time_steps, channels] arrays
        video_frame_indices: List of starting frame indices
        segment_labels: List of event labels
    """
    segment_duration = frames_per_segment / video_fps  # seconds
    
    eeg_segments = []
    video_frame_indices = []
    segment_labels = []
    
    for onset, duration, trial_type in events:
        # Calculate EEG sample indices
        eeg_start = int(onset * eeg_sample_rate)
        eeg_samples_needed = int(segment_duration * eeg_sample_rate)
        eeg_end = eeg_start + eeg_samples_needed
        
        # Skip if out of bounds
        if eeg_end > len(eeg) or eeg_start < 0:
            continue
        
        # Extract EEG segment
        eeg_seg = eeg[eeg_start:eeg_end]
        
        # Resample to target time steps
        if len(eeg_seg) != eeg_time_steps:
            # Simple linear interpolation
            indices = np.linspace(0, len(eeg_seg) - 1, eeg_time_steps)
            eeg_seg = np.array([
                np.interp(indices, np.arange(len(eeg_seg)), eeg_seg[:, ch])
                for ch in range(eeg_seg.shape[1])
            ]).T  # [eeg_time_steps, channels]
        
        eeg_segments.append(eeg_seg)
        
        # Calculate corresponding video frame index
        frame_idx = int(onset * video_fps)
        video_frame_indices.append(frame_idx)
        segment_labels.append(trial_type)
    
    return eeg_segments, video_frame_indices, segment_labels


def preprocess_natview_subject(
    subject_id: str,
    task: str,
    run: Optional[str],
    data_root: Path,
    stim_root: Path,
    output_dir: Path,
    common_channels: List[str],
    video_fps: int = 6,
    frames_per_segment: int = 6,
    resolution: int = 768,
    eeg_time_steps: int = 125,
    device: torch.device = torch.device('cuda'),
):
    """
    Preprocess a single subject for NATVIEW dataset with channel standardization
    """
    # Construct paths based on BIDS structure
    subject_dir = data_root / subject_id / 'ses-01' / 'eeg'
    
    # Task-specific file naming
    if run:
        base_name = f"{subject_id}_ses-01_task-{task}_run-{run}"
    else:
        base_name = f"{subject_id}_ses-01_task-{task}"
    
    eeg_path = subject_dir / f"{base_name}_eeg.set"
    events_path = subject_dir / f"{base_name}_events.tsv"
    
    # Video path
    video_map = {
        'dme': 'Despicable_Me_720x480_English.avi',
        'tp': 'The_Present_720x480.avi',
        'inscapes': 'Inscapes_02.avi',
    }
    video_path = stim_root / video_map[task]
    
    # Check files exist
    if not eeg_path.exists():
        print(f"  ⚠️  EEG file not found: {eeg_path}")
        return None
    if not events_path.exists():
        print(f"  ⚠️  Events file not found: {events_path}")
        return None
    if not video_path.exists():
        print(f"  ⚠️  Video file not found: {video_path}")
        return None
    
    print(f"\n{'='*70}")
    print(f"Processing {subject_id} - task-{task}" + (f" run-{run}" if run else ""))
    print(f"{'='*70}")
    
    # Load EEG data
    print(f"📊 Loading EEG from: {eeg_path}")
    eeg, eeg_sample_rate, channel_names = load_eeglab_set(eeg_path)
    
    # Standardize channels
    print(f"🔧 Standardizing to {len(common_channels)} common channels...")
    try:
        eeg = standardize_eeg_channels(eeg, channel_names, common_channels)
        print(f"  ✓ Standardized EEG shape: {eeg.shape}")
    except ValueError as e:
        print(f"  ⚠️  Channel standardization failed: {e}")
        return None
    
    # Load events
    print(f"📋 Loading events from: {events_path}")
    events = load_events(events_path)
    print(f"  ✓ Found {len(events)} events")
    
    if len(events) == 0:
        print(f"  ⚠️  No valid events found, skipping")
        return None
    
    # Load video frames
    print(f"🎬 Loading video from: {video_path}")
    video_frames = read_video_at_fps(video_path, video_fps)
    print(f"  ✓ Loaded {len(video_frames)} frames at {video_fps} FPS")
    
    # Synchronize EEG and video
    print(f"🔗 Synchronizing EEG and video...")
    eeg_segments, video_frame_indices, segment_labels = synchronize_eeg_video(
        eeg=eeg,
        events=events,
        eeg_sample_rate=int(eeg_sample_rate),
        video_fps=video_fps,
        frames_per_segment=frames_per_segment,
        eeg_time_steps=eeg_time_steps,
    )
    
    print(f"  ✓ Created {len(eeg_segments)} synchronized segments")
    
    if len(eeg_segments) == 0:
        print(f"  ⚠️  No valid segments created, skipping")
        return None
    
    # Extract corresponding video segments
    print(f"🖼️  Extracting video segments...")
    video_segments = []
    for frame_idx in tqdm(video_frame_indices, desc="Processing video"):
        segment_frames = []
        for i in range(frames_per_segment):
            idx = frame_idx + i
            if idx >= len(video_frames):
                idx = len(video_frames) - 1  # Pad with last frame
            
            frame = video_frames[idx]
            frame_resized = square_resize(frame, resolution)
            segment_frames.append(np.array(frame_resized))
        
        video_segments.append(segment_frames)
    
    print(f"  ✓ Processed {len(video_segments)} video segments")
    
    # Encode video segments with VAE
    print(f"🎨 Encoding with VAE...")
    vae, _, _, _, _ = load_diffusion_models(
        device=device,
        load_vae=True,
        load_unet=False,
        load_text_encoder=False,
    )
    
    all_latents = []
    for video_seg in tqdm(video_segments, desc="Encoding"):
        seg_latents = []
        for frame in video_seg:
            # Convert to tensor [3, H, W] in [-1, 1]
            frame_tensor = torch.from_numpy(frame).float() / 127.5 - 1.0
            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            
            # Encode with VAE
            with torch.no_grad():
                latent = vae.encode(frame_tensor).latent_dist.sample()
                latent = latent * vae.config.scaling_factor
            
            seg_latents.append(latent.cpu())
        
        seg_latents = torch.cat(seg_latents, dim=0)  # [frames, 4, 96, 96]
        all_latents.append(seg_latents)
    
    all_latents = torch.stack(all_latents, dim=0)  # [segments, frames, 4, 96, 96]
    print(f"  ✓ Latents shape: {all_latents.shape}")
    
    # Prepare output
    eeg_array = np.array([seg.T for seg in eeg_segments])  # [segments, channels, time_steps]
    video_array = np.array(video_segments)  # [segments, frames, H, W, 3]
    
    return {
        'eeg': eeg_array,
        'video': video_array,
        'latents': all_latents.numpy(),
        'metadata': {
            'subject': subject_id,
            'task': task,
            'run': run,
            'num_segments': len(eeg_segments),
            'eeg_channels': eeg_array.shape[1],
            'eeg_time_steps': eeg_array.shape[2],
            'frames_per_segment': frames_per_segment,
            'resolution': resolution,
            'video_fps': video_fps,
            'eeg_sample_rate': int(eeg_sample_rate),
        }
    }


def generate_captions_and_embeddings(
    video_frames: np.ndarray,
    context_prompt: str,
    device: torch.device,
) -> Tuple[List[str], np.ndarray]:
    """Generate BLIP captions and CLIP embeddings"""
    from PIL import Image
    
    # Load models
    print("  📥 Loading BLIP...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to(device)
    blip_model.eval()
    
    print("  📥 Loading CLIP...")
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_model.eval()
    
    # Generate captions
    all_captions = []
    all_embeddings = []
    
    for seg_idx in tqdm(range(len(video_frames)), desc="Captioning"):
        seg_frames = video_frames[seg_idx]
        seg_captions = []
        seg_embeddings = []
        
        for frame in seg_frames:
            # Generate caption
            frame_pil = Image.fromarray(frame.astype(np.uint8))
            inputs = blip_processor(frame_pil, text=context_prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                out = blip_model.generate(**inputs, max_length=50, num_beams=5)
            
            caption = blip_processor.decode(out[0], skip_special_tokens=True)
            seg_captions.append(caption)
            
            # Get CLIP embedding
            clip_inputs = clip_tokenizer(
                caption,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = clip_model(**clip_inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)  # [1, 1024]
            
            seg_embeddings.append(embedding.cpu().numpy())
        
        all_captions.append(seg_captions[0])  # Use first frame caption for segment
        all_embeddings.append(np.stack(seg_embeddings, axis=0))
    
    all_embeddings = np.stack(all_embeddings, axis=0)  # [segments, frames, 1, 1024]
    
    return all_captions, all_embeddings


def main():
    parser = argparse.ArgumentParser(description="Preprocess NATVIEW dataset")
    
    parser.add_argument('--data-root', required=True,
                       help='Path to NATVIEW data directory (contains sub-XX folders)')
    parser.add_argument('--stim-root', required=True,
                       help='Path to stimulus directory (contains video files)')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory')
    parser.add_argument('--subjects', nargs='+', default=None,
                       help='Subject IDs to process (e.g., sub-01 sub-02). If not specified, processes all.')
    parser.add_argument('--task', required=True, choices=['dme', 'tp', 'inscapes'],
                       help='Task to process')
    parser.add_argument('--runs', nargs='+', default=None,
                       help='Runs to process (e.g., 01 02). Not needed for inscapes.')
    
    # Processing parameters
    parser.add_argument('--video-fps', type=int, default=6)
    parser.add_argument('--frames-per-segment', type=int, default=6)
    parser.add_argument('--resolution', type=int, default=768)
    parser.add_argument('--eeg-time-steps', type=int, default=125)
    
    # Captioning
    parser.add_argument('--generate-captions', action='store_true', default=True,
                       help='Generate BLIP captions and CLIP embeddings')
    parser.add_argument('--context-prompt', default="a scene from an animated movie")
    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    data_root = Path(args.data_root)
    stim_root = Path(args.stim_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine subjects to process
    if args.subjects:
        subjects = args.subjects
    else:
        # Find all subjects
        subjects = sorted([d.name for d in data_root.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    
    print(f"Processing {len(subjects)} subjects: {subjects}")
    
    # Determine runs
    if args.task in ['dme', 'tp']:
        runs = args.runs if args.runs else ['01', '02']
    else:
        runs = [None]
    
    # STEP 1: Find common channels across all subjects
    print(f"\n{'='*70}")
    print("STEP 1: FINDING COMMON CHANNELS")
    print(f"{'='*70}")
    
    common_channels = find_common_channels(
        data_root=data_root,
        subjects=subjects,
        task=args.task,
        runs=runs,
    )
    
    if len(common_channels) == 0:
        print("\n❌ ERROR: No common channels found across all subjects!")
        return
    
    print(f"\n✓ Will use {len(common_channels)} common channels for all subjects")
    print(f"{'='*70}")
    
    # STEP 2: Process each subject with standardized channels
    print(f"\n{'='*70}")
    print("STEP 2: PREPROCESSING SUBJECTS")
    print(f"{'='*70}")
    
    # Save each subject immediately to avoid OOM
    temp_dir = output_dir / 'temp_subjects'
    temp_dir.mkdir(exist_ok=True)
    
    processed_files = []
    
    for subject in subjects:
        for run in runs:
            result = preprocess_natview_subject(
                subject_id=subject,
                task=args.task,
                run=run,
                data_root=data_root,
                stim_root=stim_root,
                output_dir=output_dir,
                common_channels=common_channels,
                video_fps=args.video_fps,
                frames_per_segment=args.frames_per_segment,
                resolution=args.resolution,
                eeg_time_steps=args.eeg_time_steps,
                device=device,
            )
            
            if result is not None:
                # Save immediately to disk to free memory
                run_str = f"_run-{run}" if run else ""
                temp_file = temp_dir / f"{subject}{run_str}.npz"
                
                print(f"  💾 Saving to {temp_file.name}...")
                np.savez_compressed(
                    temp_file,
                    eeg=result['eeg'],
                    video=result['video'],
                    latents=result['latents'],
                )
                processed_files.append(temp_file)
                
                # Free memory immediately
                del result
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
                print(f"  ✓ Memory cleared\n")
    
    if len(processed_files) == 0:
        print("\n⚠️  No data was successfully processed!")
        return
    
    # STEP 3: Combine all saved files
    print(f"\n{'='*70}")
    print("STEP 3: COMBINING ALL SUBJECTS")
    print(f"{'='*70}")
    print(f"Found {len(processed_files)} processed files")
    
    all_eeg = []
    all_video = []
    all_latents = []
    
    for temp_file in processed_files:
        print(f"  Loading {temp_file.name}...")
        data = np.load(temp_file)
        all_eeg.append(data['eeg'])
        all_video.append(data['video'])
        all_latents.append(data['latents'])
    
    print("\n  Concatenating arrays...")
    combined_eeg = np.concatenate(all_eeg, axis=0)
    combined_video = np.concatenate(all_video, axis=0)
    combined_latents = np.concatenate(all_latents, axis=0)
    
    # Free memory
    del all_eeg, all_video, all_latents
    import gc
    gc.collect()
    
    print(f"  Combined EEG: {combined_eeg.shape}")
    print(f"  Combined video: {combined_video.shape}")
    print(f"  Combined latents: {combined_latents.shape}")
    
    # Save combined data
    task_name = args.task
    eeg_out = output_dir / f"{task_name}_eeg.npy"
    video_out = output_dir / f"{task_name}_video_frames.npy"
    latents_out = output_dir / f"{task_name}_vae_latents_hd.npy"
    
    np.save(eeg_out, combined_eeg)
    np.save(video_out, combined_video)
    np.save(latents_out, combined_latents)
    
    print(f"\n  ✓ Saved EEG to: {eeg_out}")
    print(f"  ✓ Saved video frames to: {video_out}")
    print(f"  ✓ Saved VAE latents to: {latents_out}")
    
    # Save metadata
    metadata = {
        'task': args.task,
        'subjects': subjects,
        'runs': runs if runs != [None] else None,
        'num_segments': combined_eeg.shape[0],
        'num_subjects': len(subjects),
        'eeg_channels': combined_eeg.shape[1],
        'channel_names': common_channels,
        'eeg_time_steps': combined_eeg.shape[2],
        'frames_per_segment': args.frames_per_segment,
        'resolution': args.resolution,
        'video_fps': args.video_fps,
    }
    
    meta_out = output_dir / f"{task_name}_metadata.json"
    with open(meta_out, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Saved metadata to: {meta_out}")
    
    # Generate captions if requested
    if args.generate_captions:
        print(f"\n{'='*70}")
        print("GENERATING CAPTIONS")
        print(f"{'='*70}")
        
        captions, embeddings = generate_captions_and_embeddings(
            combined_video,
            args.context_prompt,
            device,
        )
        
        # Save captions
        captions_file = output_dir / f"{task_name}_captions_hd.json"
        with open(captions_file, 'w') as f:
            json.dump({'captions': captions}, f, indent=2)
        
        # Save CLIP embeddings
        embeddings_file = output_dir / f"{task_name}_clip_text_embeddings.npy"
        np.save(embeddings_file, embeddings)
        
        print(f"\n  ✓ Saved captions to: {captions_file}")
        print(f"  ✓ Saved CLIP embeddings to: {embeddings_file}")
    
    # Clean up temporary files
    print(f"\n🧹 Cleaning up temporary files...")
    import shutil
    shutil.rmtree(temp_dir)
    print(f"  ✓ Removed {temp_dir}")
    
    print(f"\n{'='*70}")
    print("✅ PREPROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"Total segments: {combined_eeg.shape[0]}")
    print(f"Subjects processed: {len(subjects)}")


if __name__ == "__main__":
    main()
