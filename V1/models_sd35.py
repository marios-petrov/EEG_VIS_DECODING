#!/usr/bin/env python3
"""
Model architectures for EEG2Video with SD 3.5 Medium support
- EEGAdapter: Maps EEG features to video diffusion latent space
- GLMNet: Visual cortex channel selection
- Model loading utilities for SD 2.1 and SD 3.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

# Try importing SD3 transformer
try:
    from diffusers import SD3Transformer2DModel
    SD3_AVAILABLE = True
except ImportError:
    SD3_AVAILABLE = False
    print("⚠️  SD3 not available - install with: pip install diffusers>=0.30.0")

# Fallback to UNet for SD 2.1
try:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
    UNET_AVAILABLE = True
except ImportError:
    UNET_AVAILABLE = False


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
    Projects EEG to latent space compatible with VAE-encoded videos
    Supports both SD 2.1 (4 channels) and SD 3.5 (16 channels)
    """
    def __init__(
        self,
        latent_height: int = 96,
        latent_width: int = 96,
        latent_channels: int = 4,  # 4 for SD 2.1, 16 for SD 3.5
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
    model_name: str = "stabilityai/stable-diffusion-3.5-medium",
    model_type: Literal["sd2.1", "sd3.5"] = "sd3.5",
    device: torch.device = torch.device("cuda"),
    load_vae: bool = True,
    load_transformer: bool = True,
    load_text_encoders: bool = True,
    custom_transformer_path: Optional[str] = None,
) -> Tuple:
    """
    Load diffusion model components for SD 2.1 or SD 3.5
    
    Args:
        model_name: HuggingFace model name
        model_type: "sd2.1" or "sd3.5"
        device: Device to load models on
        load_vae: Whether to load VAE
        load_transformer: Whether to load transformer/UNet
        load_text_encoders: Whether to load text encoders
        custom_transformer_path: Path to custom fine-tuned checkpoint
        
    Returns:
        Tuple of (vae, transformer/unet, text_encoders_dict, tokenizers_dict, scheduler)
    """
    vae = None
    transformer = None
    text_encoders = {}
    tokenizers = {}
    scheduler = None
    
    # Load VAE
    if load_vae:
        print("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            model_name, 
            subfolder="vae",
            torch_dtype=torch.float32,
        ).to(device)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
        print(f"  ✓ VAE latent channels: {vae.config.latent_channels}")
    
    # Load Transformer/UNet based on model type
    if load_transformer:
        if model_type == "sd3.5":
            if not SD3_AVAILABLE:
                raise ImportError("SD3 not available. Install with: pip install diffusers>=0.30.0")
            
            print("Loading SD3 Transformer...")
            transformer = SD3Transformer2DModel.from_pretrained(
                model_name,
                subfolder="transformer",
                torch_dtype=torch.float16,
            ).to(device)
            
            if custom_transformer_path:
                print(f"Loading custom transformer from {custom_transformer_path}")
                ckpt = torch.load(custom_transformer_path, map_location=device)
                if "transformer" in ckpt:
                    transformer.load_state_dict(ckpt["transformer"])
                else:
                    transformer.load_state_dict(ckpt)
            
            transformer.eval()
            for p in transformer.parameters():
                p.requires_grad = False
                
        elif model_type == "sd2.1":
            if not UNET_AVAILABLE:
                raise ImportError("UNet not available")
            
            print("Loading UNet (SD 2.1)...")
            transformer = UNet2DConditionModel.from_pretrained(
                model_name,
                subfolder="unet",
                torch_dtype=torch.float16,
            ).to(device)
            
            if custom_transformer_path:
                print(f"Loading custom UNet from {custom_transformer_path}")
                ckpt = torch.load(custom_transformer_path, map_location=device)
                if "unet" in ckpt:
                    transformer.load_state_dict(ckpt["unet"])
                else:
                    transformer.load_state_dict(ckpt)
            
            transformer.eval()
            for p in transformer.parameters():
                p.requires_grad = False
    
    # Load text encoders
    if load_text_encoders:
        print("Loading text encoders and tokenizers...")
        
        if model_type == "sd3.5":
            # SD 3.5 uses 3 text encoders
            print("  Loading CLIP-L...")
            tokenizers['clip_l'] = CLIPTokenizer.from_pretrained(
                model_name, subfolder="tokenizer"
            )
            text_encoders['clip_l'] = CLIPTextModel.from_pretrained(
                model_name, subfolder="text_encoder", torch_dtype=torch.float16
            ).to(device)
            
            print("  Loading CLIP-G...")
            tokenizers['clip_g'] = CLIPTokenizer.from_pretrained(
                model_name, subfolder="tokenizer_2"
            )
            text_encoders['clip_g'] = CLIPTextModel.from_pretrained(
                model_name, subfolder="text_encoder_2", torch_dtype=torch.float16
            ).to(device)
            
            print("  Loading T5-XXL...")
            tokenizers['t5'] = T5TokenizerFast.from_pretrained(
                model_name, subfolder="tokenizer_3"
            )
            text_encoders['t5'] = T5EncoderModel.from_pretrained(
                model_name, subfolder="text_encoder_3", torch_dtype=torch.float16
            ).to(device)
            
            # Freeze all text encoders
            for encoder in text_encoders.values():
                encoder.eval()
                for p in encoder.parameters():
                    p.requires_grad = False
                    
        elif model_type == "sd2.1":
            # SD 2.1 uses single CLIP encoder
            tokenizers['clip'] = CLIPTokenizer.from_pretrained(
                model_name, subfolder="tokenizer"
            )
            text_encoders['clip'] = CLIPTextModel.from_pretrained(
                model_name, subfolder="text_encoder", torch_dtype=torch.float16
            ).to(device)
            text_encoders['clip'].eval()
            for p in text_encoders['clip'].parameters():
                p.requires_grad = False
    
    # Load scheduler
    if model_type == "sd3.5":
        # SD 3.5 uses Flow Matching
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )
    else:
        # SD 2.1 uses DDPM
        scheduler = DDPMScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )
    
    return vae, transformer, text_encoders, tokenizers, scheduler


def get_inference_scheduler(
    model_name: str = "stabilityai/stable-diffusion-3.5-medium",
    model_type: Literal["sd2.1", "sd3.5"] = "sd3.5",
    num_inference_steps: int = 28,  # SD 3.5 default is 28
) -> object:
    """Get scheduler configured for inference"""
    if model_type == "sd3.5":
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )
        scheduler.set_timesteps(num_inference_steps)
    else:
        scheduler = DDIMScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )
        scheduler.set_timesteps(num_inference_steps)
    
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
