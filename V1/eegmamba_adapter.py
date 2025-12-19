#!/usr/bin/env python3
"""
EEG to Video Latent Adapter using EEGMamba backbone
Uses pretrained EEGMamba from HuggingFace as feature extractor
"""

import torch
import torch.nn as nn
from typing import Optional
from einops.layers.torch import Rearrange


class GLMNet(nn.Module):
    """Visual cortex channel selection"""
    def __init__(self, num_channels: int = 63, num_subjects: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.num_subjects = num_subjects
        self.channel_weights = nn.Parameter(torch.ones(num_subjects, num_channels))
        
    def forward(self, eeg: torch.Tensor, subject_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, T = eeg.shape
        if subject_ids is None:
            subject_ids = torch.zeros(B, dtype=torch.long, device=eeg.device)
        
        weights = self.channel_weights[subject_ids]
        weights = torch.sigmoid(weights)
        weighted_eeg = eeg * weights.unsqueeze(-1)
        return weighted_eeg


class EEGMambaAdapter(nn.Module):
    """
    Adapter using pretrained EEGMamba as backbone
    
    Architecture:
    1. GLMNet: Channel selection (optional)
    2. Reshape for EEGMamba: [B, C, T] → [B, C, num_segments, points_per_segment]
    3. EEGMamba: Feature extraction (pretrained, can be frozen or finetuned)
    4. MLP: Project to video latent space [B, 16, 96, 96]
    
    Input:  [B, C, T] where C=channels, T=time_steps
    Output: [B, 16, 96, 96] video latent
    """
    
    def __init__(
        self,
        # EEG input
        in_channels: int = 60,
        eeg_time_steps: int = 125,
        
        # Output latent space
        latent_height: int = 96,
        latent_width: int = 96,
        latent_channels: int = 16,  # SD 3.5
        
        # EEGMamba config
        freeze_eegmamba: bool = False,
        eegmamba_pretrained_path: str = None,
        
        # GLMNet
        use_glmnet: bool = True,
        num_subjects: int = 22,
        
        # Projection
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.eeg_time_steps = eeg_time_steps
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.latent_channels = latent_channels
        
        # 1. GLMNet for channel selection
        self.use_glmnet = use_glmnet
        if use_glmnet:
            self.glmnet = GLMNet(num_channels=in_channels, num_subjects=num_subjects)
        
        # 2. Channel projection (if needed)
        # EEGMamba expects 22 channels by default
        # We'll project from in_channels to 22 if different
        self.eegmamba_channels = 22
        if in_channels != self.eegmamba_channels:
            self.channel_proj = nn.Linear(in_channels, self.eegmamba_channels)
        else:
            self.channel_proj = nn.Identity()
        
        # 3. Load pretrained EEGMamba
        # EEGMamba expects input: [B, C=22, num_segments=4, points_per_segment=200]
        try:
            from models.eegmamba import EEGMamba
            self.eegmamba = EEGMamba()
            
            if eegmamba_pretrained_path:
                print(f"Loading pretrained EEGMamba from: {eegmamba_pretrained_path}")
                state_dict = torch.load(eegmamba_pretrained_path, map_location='cpu')
                self.eegmamba.load_state_dict(state_dict, strict=False)
                print("✓ EEGMamba loaded successfully")
            
            # Remove original projection head
            self.eegmamba.proj_out = nn.Identity()
            
            # Freeze if requested
            if freeze_eegmamba:
                for param in self.eegmamba.parameters():
                    param.requires_grad = False
                print("✓ EEGMamba frozen")
            else:
                print("✓ EEGMamba will be finetuned")
                
        except ImportError:
            raise ImportError(
                "EEGMamba model not found. Please clone the repo:\n"
                "git clone https://github.com/wjq-learning/EEGMamba.git\n"
                "And add 'models' folder to your Python path."
            )
        
        # Calculate EEGMamba output dimensions
        # EEGMamba outputs: [B, C=22, num_segments=4, points_per_segment=200]
        self.num_segments = 4
        self.points_per_segment = 200
        eegmamba_output_dim = self.eegmamba_channels * self.num_segments * self.points_per_segment
        
        # 4. Projection head to video latent space
        latent_dim = latent_channels * latent_height * latent_width
        
        self.projection = nn.Sequential(
            Rearrange('b c s p -> b (c s p)'),  # Flatten
            nn.LayerNorm(eegmamba_output_dim),
            nn.Linear(eegmamba_output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        
    def prepare_eeg_for_mamba(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Convert raw EEG to EEGMamba format
        Input:  [B, C, T] where T is total time steps
        Output: [B, 22, 4, 200] (EEGMamba format)
        """
        B, C, T = eeg.shape
        
        # Project channels if needed
        if C != self.eegmamba_channels:
            # [B, C, T] -> [B, T, C] -> [B, T, 22] -> [B, 22, T]
            eeg = eeg.transpose(1, 2)
            eeg = self.channel_proj(eeg)
            eeg = eeg.transpose(1, 2)
        
        # Reshape to [B, 22, num_segments, points_per_segment]
        # We need to split T into segments
        target_length = self.num_segments * self.points_per_segment  # 4 * 200 = 800
        
        if T < target_length:
            # Pad if too short
            pad_length = target_length - T
            eeg = torch.nn.functional.pad(eeg, (0, pad_length), mode='replicate')
        elif T > target_length:
            # Interpolate if too long
            eeg = torch.nn.functional.interpolate(
                eeg, size=target_length, mode='linear', align_corners=False
            )
        
        # Reshape to [B, 22, 4, 200]
        eeg = eeg.reshape(B, self.eegmamba_channels, self.num_segments, self.points_per_segment)
        
        return eeg
        
    def forward(
        self, 
        eeg: torch.Tensor, 
        subject_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            eeg: [B, C, T] EEG signals
            subject_ids: [B] subject indices for GLMNet
        Returns:
            latent: [B, 16, 96, 96] video latent
        """
        # 1. GLMNet channel selection
        if self.use_glmnet:
            eeg = self.glmnet(eeg, subject_ids)
        
        # 2. Prepare for EEGMamba
        eeg_mamba_format = self.prepare_eeg_for_mamba(eeg)
        
        # 3. EEGMamba feature extraction
        features = self.eegmamba(eeg_mamba_format)  # [B, 22, 4, 200]
        
        # 4. Project to video latent space
        latent_flat = self.projection(features)  # [B, C*H*W]
        
        # Reshape to 4D latent
        latent = latent_flat.view(
            -1,
            self.latent_channels,
            self.latent_height,
            self.latent_width
        )
        
        return latent


if __name__ == "__main__":
    # Test the adapter
    print("Testing EEGMambaAdapter...")
    
    adapter = EEGMambaAdapter(
        in_channels=60,
        eeg_time_steps=125,
        latent_channels=16,
        latent_height=96,
        latent_width=96,
        use_glmnet=True,
        num_subjects=22,
        freeze_eegmamba=False,
    )
    
    # Mock input
    batch_size = 4
    eeg = torch.randn(batch_size, 60, 125)
    subject_ids = torch.randint(0, 22, (batch_size,))
    
    # Forward pass
    latent = adapter(eeg, subject_ids)
    
    print(f"\n✓ Forward pass successful!")
    print(f"  Input EEG: {eeg.shape}")
    print(f"  Output latent: {latent.shape}")
    print(f"  Expected: torch.Size([4, 16, 96, 96])")
    
    # Count parameters
    total_params = sum(p.numel() for p in adapter.parameters())
    trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    
    print(f"\n📊 Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
