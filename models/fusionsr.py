"""
FusionSR-v5 — Novel Hybrid Architecture for 4× Super-Resolution.

Architecture (~14.0M params):

    Stage 1 — Shallow Extractor:
        Single Conv2d(3→162, 3×3).

    Stage 2 — Deep Feature Extraction (8 residual groups):
        Each group: HFEB → MultiScaleWindowGroup(4 HybridSwinBlocks) → Conv → skip
        TDCA inserted after groups 2, 4, 6, 8 for global self-similarity.
        Long skip over entire body.

    Stage 3 — Residual Reconstruction:
        Bicubic upsampled input provides coarse HR estimate.
        Progressive 2×+2× PixelShuffle produces the learned residual.
        Output = bicubic + residual.

Novel components (vs SwinIR):
    - Multi-Scale Window Attention (ws=4 + ws=8) — no SR model uses mixed sizes
    - HFEB (CRAFT-inspired) — explicit HF feature emphasis, replaces RCAB
    - Hybrid channel attention inside each SwinBlock (HAT-inspired)
    - Token Dictionary Cross-Attention (ATD-inspired) — global O(N×K) attention

Safe mode flags:
    use_hfeb, use_hybrid_ca, use_mswa, use_tdca — each disables one novel
    component and falls back to a standard/safe equivalent.

Changes from v4:
    - channels: 180 → 162
    - num_groups: 6 → 8
    - RCAB×6 per group → HFEB (58× cheaper)
    - CAB → removed (channel attention now inside each block)
    - SwinBlockPair → MultiScaleWindowGroup (4 blocks, mixed ws=4+8)
    - OCA → removed (TDCA handles global context)
    - window_size: 16 → mixed 4+8
    - Reconstruction: progressive → residual + progressive PS(2)+PS(2)
    - _pad_to_window: hardcoded to LCM=8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from models.blocks import ResidualGroup, TokenDictionaryCrossAttention


class FusionSR(nn.Module):
    """FusionSR-v5 generator network."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 162,
        num_groups: int = 8,
        num_heads: int = 6,
        scale: int = 4,
        ffn_expansion: float = 2.0,
        # Safe mode flags — each disables one novel component
        use_hfeb: bool = True,
        use_hybrid_ca: bool = True,
        use_mswa: bool = True,
        use_tdca: bool = True,
        # TDCA config
        tdca_num_tokens: int = 64,
        tdca_interval: int = 2,  # insert TDCA every N groups
        # HFEB config
        hf_scale_init: float = 0.01,
        # Checkpointing
        use_checkpoint: bool = False,
        # Legacy kwargs (ignored, for backward compat with evaluate/inference configs)
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.scale = scale
        # Hardcoded to LCM of all window sizes (4, 8) = 8
        # This ensures _pad_to_window works correctly at inference on
        # arbitrary image sizes, regardless of individual block window sizes.
        self.window_size = 8

        self.use_tdca = use_tdca
        self.tdca_interval = tdca_interval

        # ── Stage 1 — shallow feature extraction ──
        self.shallow = nn.Conv2d(in_channels, channels, 3, padding=1, bias=True)

        # ── Stage 2 — deep feature extraction ──
        self.groups = nn.ModuleList()
        self.tdca_layers = nn.ModuleDict()

        for i in range(num_groups):
            self.groups.append(
                ResidualGroup(
                    channels=channels,
                    num_heads=num_heads,
                    ffn_expansion=ffn_expansion,
                    use_hfeb=use_hfeb,
                    use_mswa=use_mswa,
                    use_hybrid_ca=use_hybrid_ca,
                    hf_scale_init=hf_scale_init,
                )
            )

            # TDCA after every tdca_interval groups (e.g., groups 1, 3, 5, 7 = 0-indexed)
            if use_tdca and (i + 1) % tdca_interval == 0:
                self.tdca_layers[str(i)] = TokenDictionaryCrossAttention(
                    channels=channels,
                    num_tokens=tdca_num_tokens,
                    num_heads=num_heads,  # use same number of heads to ensure divisibility (162 % 6 == 0)
                )

        self.body_conv = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

        # ── Stage 3 — residual reconstruction (progressive 2×+2× PS) ──
        # The model learns the HIGH-FREQUENCY RESIDUAL over bicubic upsampling.
        # This makes training easier — the loss starts low because bicubic
        # is already a decent baseline, and the body focuses on learning the
        # missing high-frequency detail rather than the entire image.
        self.reconstruction = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            # first 2× upscale
            nn.Conv2d(channels, channels * 4, 3, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.GELU(),
            # second 2× upscale → total 4×
            nn.Conv2d(channels, channels * 4, 3, padding=1, bias=True),
            nn.PixelShuffle(2),
            # final refinement → RGB residual
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, out_channels, 3, padding=1, bias=True),
        )

        # Print parameter count on init
        total_params = sum(p.numel() for p in self.parameters())
        print(f"FusionSR-v5 initialized: {total_params / 1e6:.2f}M parameters")
        if kwargs:
            print(f"  (ignored legacy kwargs: {list(kwargs.keys())})")

    def _pad_to_window(self, x: torch.Tensor):
        """Pad spatial dims to multiple of 8 (LCM of all window sizes: 4, 8).

        Hardcoded to 8 regardless of individual block window sizes.
        This ensures correct behavior at inference on arbitrary image sizes.
        """
        _, _, H, W = x.shape
        ws = 8  # LCM of 4 and 8
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, H, W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bicubic coarse estimate (no parameters — just interpolation)
        coarse_hr = F.interpolate(
            x, scale_factor=self.scale, mode="bicubic",
            align_corners=False, antialias=True,
        )

        # Pad for window compatibility
        x, orig_H, orig_W = self._pad_to_window(x)

        # Stage 1 — shallow features
        shallow = self.shallow(x)

        # Stage 2 — deep feature extraction with TDCA
        deep = shallow
        for i, group in enumerate(self.groups):
            if self.use_checkpoint and deep.requires_grad:
                deep = checkpoint.checkpoint(group, deep, use_reentrant=False)
            else:
                deep = group(deep)
                
            # Apply TDCA after designated groups
            if self.use_tdca and str(i) in self.tdca_layers:
                tdca = self.tdca_layers[str(i)]
                if self.use_checkpoint and deep.requires_grad:
                    deep = checkpoint.checkpoint(tdca, deep, use_reentrant=False)
                else:
                    deep = tdca(deep)

        deep = self.body_conv(deep)

        # Long skip connection
        fused = deep + shallow

        # Stage 3 — residual reconstruction
        residual_hr = self.reconstruction(fused)

        # Crop to original HR size (accounting for padding and scale)
        residual_hr = residual_hr[:, :, : orig_H * self.scale, : orig_W * self.scale]

        # Crop coarse_hr to match (it was computed before padding)
        coarse_hr = coarse_hr[:, :, : orig_H * self.scale, : orig_W * self.scale]

        # Output = bicubic baseline + learned high-frequency residual
        return coarse_hr + residual_hr


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
