#!/usr/bin/env python3
"""
Model architectures for EEG2Video
- EEGAdapter: Maps EEG features to video diffusion latent space
- GLMNet: Visual cortex channel selection
- Model loading utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


class GLMNet(nn.Module):
    """
    GLMNet for visual cortex channel selection
    From: "Reconstructing Perceived Images from Brain Activity by Visually-Guided Cognitive Representation and Adversarial Learning"
    """
    def __init__(self, num_channels: int = 63, num_subjects: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.num_subjects = num_subjects
        
        # Per-subject channel selection weights
        self.channel_weights = nn.Parameter(torch.ones(num_subjects, num_channels))
        
    def forward(self, eeg: torch.Tensor, subject_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            eeg: [B, C, T] EEG signals
            subject_ids: [B] subject indices (0 to num_subjects-1)
        Returns:
            weighted_eeg: [B, C, T] with channel selection applied
        """
        B, C, T = eeg.shape
        
        if subject_ids is None:
            subject_ids = torch.zeros(B, dtype=torch.long, device=eeg.device)
        
        # Get weights for each sample's subject
        weights = self.channel_weights[subject_ids]  # [B, C]
        weights = torch.sigmoid(weights)  # Normalize to [0, 1]
        
        # Apply channel selection
        weighted_eeg = eeg * weights.unsqueeze(-1)  # [B, C, T]
        
        return weighted_eeg
    
    def get_selected_channels(self, subject_id: int = 0, threshold: float = 0.5) -> torch.Tensor:
        """Get indices of selected channels for a subject"""
        weights = torch.sigmoid(self.channel_weights[subject_id])
        return (weights > threshold).nonzero(as_tuple=True)[0]


class EEGAdapter(nn.Module):
    """
    Adapter network to map EEG features to video diffusion latent space
    Architecture: CNN + Transformer + MLP projection
    """
    def __init__(
        self,
        in_channels: int = 63,
        eeg_time_steps: int = 125,
        latent_dim: int = 512,
        hidden_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        use_glmnet: bool = True,
        num_subjects: int = 1,
    ):
        super().__init__()
        
        self.use_glmnet = use_glmnet
        if use_glmnet:
            self.glmnet = GLMNet(num_channels=in_channels, num_subjects=num_subjects)
        
        # Convolutional feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(256, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        # Calculate sequence length after pooling
        self.seq_len = eeg_time_steps // 4
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.seq_len, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection to latent space
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        
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
            latent: [B, latent_dim] latent representation
        """
        # Apply GLMNet channel selection
        if self.use_glmnet:
            eeg = self.glmnet(eeg, subject_ids)
        
        # CNN feature extraction
        x = self.conv_layers(eeg)  # [B, hidden_dim, seq_len]
        x = x.transpose(1, 2)  # [B, seq_len, hidden_dim]
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Transformer encoding
        x = self.transformer(x)  # [B, seq_len, hidden_dim]
        
        # Global average pooling
        x = x.mean(dim=1)  # [B, hidden_dim]
        
        # Project to latent space
        latent = self.output_proj(x)  # [B, latent_dim]
        
        return latent


class VideoLatentAdapter(EEGAdapter):
    """
    Specialized adapter for video diffusion models
    Projects EEG to 4D latent space compatible with VAE-encoded videos
    """
    def __init__(
        self,
        latent_height: int = 96,
        latent_width: int = 96,
        latent_channels: int = 4,
        **kwargs
    ):
        # Calculate required latent_dim
        latent_dim = latent_channels * latent_height * latent_width
        super().__init__(latent_dim=latent_dim, **kwargs)
        
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.latent_channels = latent_channels
        
    def forward(
        self, 
        eeg: torch.Tensor, 
        subject_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            eeg: [B, C, T] EEG signals
            subject_ids: [B] subject indices
        Returns:
            latent: [B, latent_channels, latent_height, latent_width]
        """
        # Get flattened latent
        latent_flat = super().forward(eeg, subject_ids)  # [B, C*H*W]
        
        # Reshape to 4D
        latent = latent_flat.view(
            -1, 
            self.latent_channels, 
            self.latent_height, 
            self.latent_width
        )
        
        return latent


def load_diffusion_models(
    model_name: str = "stabilityai/stable-diffusion-2-1",
    device: torch.device = torch.device("cuda"),
    load_vae: bool = True,
    load_unet: bool = True,
    load_text_encoder: bool = True,
    custom_unet_path: Optional[str] = None,
) -> Tuple:
    """
    Load diffusion model components
    
    Args:
        model_name: HuggingFace model name
        device: Device to load models on
        load_vae: Whether to load VAE
        load_unet: Whether to load UNet
        load_text_encoder: Whether to load text encoder
        custom_unet_path: Path to custom fine-tuned UNet checkpoint
        
    Returns:
        Tuple of (vae, unet, text_encoder, tokenizer, scheduler)
        (None for components not loaded)
    """
    vae = None
    unet = None
    text_encoder = None
    tokenizer = None
    scheduler = None
    
    if load_vae:
        print("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            model_name, 
            subfolder="vae"
        ).to(device)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
    
    if load_unet:
        print("Loading UNet...")
        unet = UNet2DConditionModel.from_pretrained(
            model_name,
            subfolder="unet"
        ).to(device)
        
        # Load custom checkpoint if provided
        if custom_unet_path:
            print(f"Loading custom UNet from {custom_unet_path}")
            ckpt = torch.load(custom_unet_path, map_location=device)
            if "unet" in ckpt:
                unet.load_state_dict(ckpt["unet"])
            else:
                unet.load_state_dict(ckpt)
        
        unet.eval()
        for p in unet.parameters():
            p.requires_grad = False
    
    if load_text_encoder:
        print("Loading text encoder and tokenizer...")
        tokenizer = CLIPTokenizer.from_pretrained(
            model_name,
            subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            model_name,
            subfolder="text_encoder"
        ).to(device)
        text_encoder.eval()
        for p in text_encoder.parameters():
            p.requires_grad = False
    
    # Always load scheduler
    scheduler = DDPMScheduler.from_pretrained(
        model_name,
        subfolder="scheduler"
    )
    
    return vae, unet, text_encoder, tokenizer, scheduler


def get_inference_scheduler(
    model_name: str = "stabilityai/stable-diffusion-2-1",
    scheduler_type: str = "ddim",
    num_inference_steps: int = 50,
) -> DDIMScheduler:
    """Get scheduler configured for inference"""
    if scheduler_type == "ddim":
        scheduler = DDIMScheduler.from_pretrained(
            model_name,
            subfolder="scheduler"
        )
        scheduler.set_timesteps(num_inference_steps)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, mu=0.999, device='cpu'):
        self.mu = mu
        self.device = device
        # Store shadow weights on CPU to save memory
        self.shadow = {
            k: v.detach().clone().cpu() 
            for k, v in model.state_dict().items() 
            if v.dtype.is_floating_point
        }
        self.backup = {}
    
    @torch.no_grad()
    def update(self, model):
        """Update EMA parameters"""
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                v_cpu = v.detach().cpu()
                self.shadow[k].mul_(self.mu).add_(v_cpu, alpha=1.0 - self.mu)
    
    @torch.no_grad()
    def store(self, model):
        """Store current parameters and load EMA weights"""
        self.backup = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
        model_device = next(model.parameters()).device
        shadow_on_device = {k: v.to(model_device) for k, v in self.shadow.items()}
        model.load_state_dict({**model.state_dict(), **shadow_on_device}, strict=False)
    
    @torch.no_grad()
    def restore(self, model):
        """Restore original parameters"""
        if self.backup:
            model_device = next(model.parameters()).device
            backup_on_device = {k: v.to(model_device) for k, v in self.backup.items()}
            model.load_state_dict(backup_on_device, strict=False)
