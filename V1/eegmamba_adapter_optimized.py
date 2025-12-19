#!/usr/bin/env python3
"""
EEGMamba Adapter for Stable Diffusion 3.5

Converts EEG signals to SD 3.5 VAE latent space using Mamba (State Space Model) architecture.
Supports both NATVIEW (128 ch, 250 samples) and SEED-DV (62 ch, 400 samples) datasets.

Architecture:
    EEG Input → Patch Embedding → Mamba Blocks → Cross-Attention (text) → Latent Decoder → SD 3.5 Latent

Key Features:
    - Configurable EEG channels and samples for different datasets
    - Mamba blocks for efficient sequential modeling
    - Optional text conditioning via cross-attention
    - Outputs 16-channel latent at 96x96 (SD 3.5 format for 768x768 images)

Usage:
    model = EEGMambaAdapter(
        eeg_channels=62,      # 62 for SEED-DV, 128 for NATVIEW
        eeg_samples=400,      # 400 for SEED-DV (2s@200Hz), 250 for NATVIEW (0.5s@500Hz)
        d_model=512,
        n_layers=4,
        latent_channels=16,
        latent_height=96,
        latent_width=96,
        text_embed_dim=2048,  # T5 embedding dim for SD 3.5
    )
    
    latent = model(eeg, text_embedding)  # (B, 16, 96, 96)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange, repeat


# ============================================================================
# Mamba Components
# ============================================================================

class MambaBlock(nn.Module):
    """
    Simplified Mamba block for EEG processing.
    
    Based on the Mamba architecture (Gu & Dao, 2023) but implemented in pure PyTorch
    for compatibility. Uses selective state space with input-dependent parameters.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution for local context
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, dt
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # Initialize dt bias for stability
        dt_init_std = 0.001
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # A parameter (diagonal state matrix)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Layer norm and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) input tensor
        Returns:
            (B, L, D) output tensor
        """
        B, L, D = x.shape
        
        # Residual
        residual = x
        x = self.norm(x)
        
        # Input projection and split
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)
        
        # Convolution
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :L]  # Causal padding
        x = rearrange(x, 'b d l -> b l d')
        
        # Activation
        x = F.silu(x)
        
        # SSM
        y = self.ssm(x)
        
        # Gate and output
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)
        
        return y + residual
    
    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Selective State Space Model"""
        B, L, D = x.shape
        
        # Compute input-dependent parameters
        x_proj = self.x_proj(x)  # (B, L, d_state*2 + 1)
        
        # Split into B, C, dt
        B_param = x_proj[:, :, :self.d_state]  # (B, L, d_state)
        C_param = x_proj[:, :, self.d_state:2*self.d_state]  # (B, L, d_state)
        dt = x_proj[:, :, -1:]  # (B, L, 1)
        
        # Project dt to d_inner dimension
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)
        
        # Get A (negative for stability)
        A = -torch.exp(self.A_log)  # (d_state,)
        
        # Discretize A and B
        # A_bar = exp(dt * A)
        # B_bar = dt * B
        dA = torch.exp(dt.unsqueeze(-1) * A)  # (B, L, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B_param.unsqueeze(2)  # (B, L, d_inner, d_state)
        
        # Scan (simplified - for efficiency, use parallel scan in production)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        
        for i in range(L):
            h = dA[:, i] * h + dB[:, i] * x[:, i].unsqueeze(-1)
            y = (h * C_param[:, i].unsqueeze(1)).sum(-1)  # (B, d_inner)
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        
        # Add skip connection
        y = y + x * self.D
        
        return y


class BidirectionalMamba(nn.Module):
    """Bidirectional Mamba for capturing both forward and backward dependencies."""
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.forward_mamba = MambaBlock(d_model, d_state, d_conv, expand, dropout)
        self.backward_mamba = MambaBlock(d_model, d_state, d_conv, expand, dropout)
        self.merge = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass
        fwd = self.forward_mamba(x)
        
        # Backward pass (flip, process, flip back)
        bwd = self.backward_mamba(x.flip(1)).flip(1)
        
        # Merge
        merged = torch.cat([fwd, bwd], dim=-1)
        out = self.merge(merged)
        
        return self.norm(out)


# ============================================================================
# Cross-Attention for Text Conditioning
# ============================================================================

class CrossAttention(nn.Module):
    """Cross-attention layer for conditioning on text embeddings."""
    
    def __init__(self, d_model: int, d_context: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_context, d_model)
        self.v_proj = nn.Linear(d_context, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) query tensor
            context: (B, D_context) or (B, L_ctx, D_context) context tensor
        """
        B, L, D = x.shape
        
        # Handle 1D context (expand to sequence)
        if context.dim() == 2:
            context = context.unsqueeze(1)  # (B, 1, D_context)
        
        # Pre-norm
        x_norm = self.norm1(x)
        
        # Project Q, K, V
        q = self.q_proj(x_norm)  # (B, L, D)
        k = self.k_proj(context)  # (B, L_ctx, D)
        v = self.v_proj(context)  # (B, L_ctx, D)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.n_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.n_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.n_heads)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.out_proj(out)
        
        # Residual + FFN
        x = x + self.dropout(out)
        x = x + self.ffn(self.norm2(x))
        
        return x


# ============================================================================
# EEG Patch Embedding
# ============================================================================

class EEGPatchEmbedding(nn.Module):
    """
    Convert EEG signals to patch embeddings.
    
    Treats each temporal window across all channels as a patch.
    """
    
    def __init__(
        self,
        eeg_channels: int,
        eeg_samples: int,
        d_model: int,
        patch_size: int = 25,  # Temporal patch size
    ):
        super().__init__()
        
        self.eeg_channels = eeg_channels
        self.eeg_samples = eeg_samples
        self.patch_size = patch_size
        self.n_patches = eeg_samples // patch_size
        
        # Patch embedding: (channels * patch_size) -> d_model
        self.proj = nn.Linear(eeg_channels * patch_size, d_model)
        
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        
        # Channel embedding (learnable per-channel weights)
        self.channel_embed = nn.Parameter(torch.randn(1, eeg_channels, 1) * 0.02)
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) EEG tensor where C=channels, T=samples
        Returns:
            (B, N, D) patch embeddings where N=n_patches
        """
        B, C, T = x.shape
        
        # Apply channel weighting
        x = x * F.softmax(self.channel_embed, dim=1)
        
        # Reshape into patches: (B, C, T) -> (B, N, C*P)
        x = x[:, :, :self.n_patches * self.patch_size]  # Truncate to fit
        x = rearrange(x, 'b c (n p) -> b n (c p)', p=self.patch_size)
        
        # Project to d_model
        x = self.proj(x)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        return self.norm(x)


# ============================================================================
# Latent Decoder
# ============================================================================

class LatentDecoder(nn.Module):
    """
    Decode Mamba features to SD 3.5 VAE latent space.
    
    Uses progressive upsampling with residual blocks.
    """
    
    def __init__(
        self,
        d_model: int,
        latent_channels: int = 16,
        latent_height: int = 96,
        latent_width: int = 96,
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.latent_height = latent_height
        self.latent_width = latent_width
        
        # Initial projection
        self.init_size = 6  # Start from 6x6
        self.init_proj = nn.Linear(d_model, 512 * self.init_size * self.init_size)
        
        # Upsampling blocks: 6 -> 12 -> 24 -> 48 -> 96
        self.up_blocks = nn.ModuleList([
            self._make_up_block(512, 256),   # 6 -> 12
            self._make_up_block(256, 128),   # 12 -> 24
            self._make_up_block(128, 64),    # 24 -> 48
            self._make_up_block(64, 32),     # 48 -> 96
        ])
        
        # Final projection to latent channels
        self.final = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, latent_channels, kernel_size=3, padding=1),
        )
    
    def _make_up_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) pooled features
        Returns:
            (B, C, H, W) latent tensor
        """
        B = x.shape[0]
        
        # Project and reshape
        x = self.init_proj(x)
        x = x.view(B, 512, self.init_size, self.init_size)
        
        # Upsample
        for block in self.up_blocks:
            x = block(x)
        
        # Final projection
        x = self.final(x)
        
        return x


# ============================================================================
# Main Model
# ============================================================================

class EEGMambaAdapter(nn.Module):
    """
    EEGMamba Adapter for Stable Diffusion 3.5
    
    Converts EEG signals to SD 3.5 VAE latent space using Mamba architecture
    with optional text conditioning.
    
    Args:
        eeg_channels: Number of EEG channels (62 for SEED-DV, 128 for NATVIEW)
        eeg_samples: Number of time samples (400 for SEED-DV, 250 for NATVIEW)
        d_model: Model dimension
        n_layers: Number of Mamba layers
        d_state: SSM state dimension
        d_conv: Convolution kernel size in Mamba
        expand: Expansion factor for Mamba inner dimension
        n_heads: Number of attention heads for cross-attention
        latent_channels: Output latent channels (16 for SD 3.5)
        latent_height: Output latent height (96 for 768px images)
        latent_width: Output latent width (96 for 768px images)
        text_embed_dim: Text embedding dimension (2048 for T5-XXL)
        dropout: Dropout rate
        patch_size: Temporal patch size for EEG embedding
    """
    
    def __init__(
        self,
        eeg_channels: int = 128,
        eeg_samples: int = 250,
        d_model: int = 512,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_heads: int = 8,
        latent_channels: int = 16,
        latent_height: int = 96,
        latent_width: int = 96,
        text_embed_dim: int = 2048,
        dropout: float = 0.1,
        patch_size: int = 25,
    ):
        super().__init__()
        
        self.eeg_channels = eeg_channels
        self.eeg_samples = eeg_samples
        self.d_model = d_model
        
        # Adjust patch size if needed
        if eeg_samples % patch_size != 0:
            # Find a suitable patch size
            for ps in [25, 20, 16, 10, 8, 5, 4, 2, 1]:
                if eeg_samples % ps == 0:
                    patch_size = ps
                    break
        
        # EEG embedding
        self.eeg_embed = EEGPatchEmbedding(
            eeg_channels=eeg_channels,
            eeg_samples=eeg_samples,
            d_model=d_model,
            patch_size=patch_size,
        )
        
        # Mamba layers (bidirectional for better context)
        self.mamba_layers = nn.ModuleList([
            BidirectionalMamba(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        
        # Text conditioning via cross-attention
        self.text_proj = nn.Linear(text_embed_dim, d_model)
        self.cross_attn_layers = nn.ModuleList([
            CrossAttention(d_model, d_model, n_heads, dropout)
            for _ in range(n_layers // 2)  # Cross-attention every 2 Mamba layers
        ])
        
        # Global pooling and feature refinement
        self.pool_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Latent decoder
        self.latent_decoder = LatentDecoder(
            d_model=d_model,
            latent_channels=latent_channels,
            latent_height=latent_height,
            latent_width=latent_width,
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(
        self,
        eeg: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            eeg: (B, C, T) EEG tensor
            text_embed: (B, D_text) optional text embedding
        
        Returns:
            (B, latent_channels, latent_height, latent_width) latent tensor
        """
        B = eeg.shape[0]
        
        # Embed EEG
        x = self.eeg_embed(eeg)  # (B, N, D)
        
        # Process text embedding
        if text_embed is not None:
            text_feat = self.text_proj(text_embed)  # (B, D)
        else:
            text_feat = torch.zeros(B, self.d_model, device=eeg.device, dtype=eeg.dtype)
        
        # Apply Mamba layers with cross-attention
        cross_attn_idx = 0
        for i, mamba_layer in enumerate(self.mamba_layers):
            x = mamba_layer(x)
            
            # Apply cross-attention every 2 layers
            if (i + 1) % 2 == 0 and cross_attn_idx < len(self.cross_attn_layers):
                x = self.cross_attn_layers[cross_attn_idx](x, text_feat)
                cross_attn_idx += 1
        
        # Global average pooling
        x = x.mean(dim=1)  # (B, D)
        
        # Feature refinement
        x = self.pool_proj(x)
        
        # Decode to latent
        latent = self.latent_decoder(x)  # (B, C, H, W)
        
        return latent
    
    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Utility Functions
# ============================================================================

def create_eegmamba_for_natview() -> EEGMambaAdapter:
    """Create EEGMamba configured for NATVIEW dataset (128ch, 500Hz, 0.5s)"""
    return EEGMambaAdapter(
        eeg_channels=128,
        eeg_samples=250,
        d_model=512,
        n_layers=4,
        latent_channels=16,
        latent_height=96,
        latent_width=96,
        text_embed_dim=2048,
        patch_size=25,
    )


def create_eegmamba_for_seed_dv() -> EEGMambaAdapter:
    """Create EEGMamba configured for SEED-DV dataset (62ch, 200Hz, 2s)"""
    return EEGMambaAdapter(
        eeg_channels=62,
        eeg_samples=400,
        d_model=512,
        n_layers=4,
        latent_channels=16,
        latent_height=96,
        latent_width=96,
        text_embed_dim=2048,
        patch_size=25,
    )


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing EEGMamba Adapter...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test NATVIEW configuration
    print("\n--- NATVIEW Configuration ---")
    model_natview = create_eegmamba_for_natview().to(device)
    print(f"Parameters: {model_natview.get_num_params():,}")
    
    eeg_natview = torch.randn(2, 128, 250).to(device)
    text_natview = torch.randn(2, 2048).to(device)
    
    with torch.no_grad():
        out_natview = model_natview(eeg_natview, text_natview)
    print(f"Input shape: {eeg_natview.shape}")
    print(f"Output shape: {out_natview.shape}")
    assert out_natview.shape == (2, 16, 96, 96), f"Expected (2, 16, 96, 96), got {out_natview.shape}"
    print("✓ NATVIEW test passed")
    
    # Test SEED-DV configuration
    print("\n--- SEED-DV Configuration ---")
    model_seed = create_eegmamba_for_seed_dv().to(device)
    print(f"Parameters: {model_seed.get_num_params():,}")
    
    eeg_seed = torch.randn(2, 62, 400).to(device)
    text_seed = torch.randn(2, 2048).to(device)
    
    with torch.no_grad():
        out_seed = model_seed(eeg_seed, text_seed)
    print(f"Input shape: {eeg_seed.shape}")
    print(f"Output shape: {out_seed.shape}")
    assert out_seed.shape == (2, 16, 96, 96), f"Expected (2, 16, 96, 96), got {out_seed.shape}"
    print("✓ SEED-DV test passed")
    
    # Test without text conditioning
    print("\n--- No Text Conditioning ---")
    with torch.no_grad():
        out_no_text = model_seed(eeg_seed)
    print(f"Output shape (no text): {out_no_text.shape}")
    assert out_no_text.shape == (2, 16, 96, 96)
    print("✓ No-text test passed")
    
    # Test mixed precision
    print("\n--- Mixed Precision Test ---")
    with torch.cuda.amp.autocast():
        out_amp = model_seed(eeg_seed, text_seed)
    print(f"AMP output dtype: {out_amp.dtype}")
    print("✓ Mixed precision test passed")
    
    print("\n✅ All tests passed!")
