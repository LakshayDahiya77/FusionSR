"""
FusionSR-v3 — Hybrid CNN-Transformer for 4× Super-Resolution.

Architecture (~8.6M params):

    Stage 1 — Shallow Extractor:
        Single Conv2d(3→96, 3×3). Replaces v2's dual-path extractor
        whose secondary path (Path B) contributed only 12% weight.

    Stage 2 — Deep Feature Extraction (6 residual groups):
        Each group: RCAB×6 → ChannelAttentionBridge → SwinBlockPair(GDFN) → Conv → skip
        Long skip over entire body.

    Stage 3 — Progressive Reconstruction:
        Two-stage PixelShuffle (2×→2× = 4×) instead of v2's single 4× step.
        Conv → PS(2) → GELU → Conv → PS(2) → Conv → GELU → Conv(→RGB)

Changes from v2:
    - Simplified shallow extractor (single conv vs dual-path)
    - GDFN replaces Linear FFN in Swin blocks
    - ChannelAttentionBridge between RCAB and Swin stages
    - Progressive 2×+2× reconstruction
"""

import torch
import torch.nn as nn
from models.blocks import ResidualGroup


class FusionSR(nn.Module):
    """FusionSR-v3 generator network."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 96,
        num_groups: int = 6,
        num_rcab: int = 6,
        window_size: int = 8,
        num_heads: int = 4,
        scale: int = 4,
        ffn_expansion: float = 2.0,
    ):
        super().__init__()
        self.scale = scale
        self.window_size = window_size

        # ── Stage 1 — shallow feature extraction ──
        # single conv replaces v2's dual-path (Path A dominated at 88%)
        self.shallow = nn.Conv2d(in_channels, channels, 3, padding=1, bias=True)

        # ── Stage 2 — deep feature extraction ──
        self.body = nn.Sequential(
            *[
                ResidualGroup(
                    channels, window_size, num_heads, num_rcab, ffn_expansion
                )
                for _ in range(num_groups)
            ]
        )
        self.body_conv = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

        # ── Stage 3 — progressive reconstruction (2× + 2× = 4×) ──
        # two-stage PixelShuffle avoids v2's hard 4× mapping
        self.reconstruction = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            # first 2× upscale
            nn.Conv2d(channels, channels * 4, 3, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.GELU(),
            # second 2× upscale → total 4×
            nn.Conv2d(channels, channels * 4, 3, padding=1, bias=True),
            nn.PixelShuffle(2),
            # final refinement
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, out_channels, 3, padding=1, bias=True),
        )

    def _pad_to_window(self, x: torch.Tensor):
        """Pad spatial dims to multiple of window_size."""
        _, _, H, W = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, H, W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pad for Swin window compatibility
        x, orig_H, orig_W = self._pad_to_window(x)

        # Stage 1
        shallow = self.shallow(x)

        # Stage 2
        deep = self.body(shallow)
        deep = self.body_conv(deep)

        # long skip connection
        fused = deep + shallow

        # Stage 3
        out = self.reconstruction(fused)

        # crop to original HR size (accounting for 4× scale)
        return out[:, :, : orig_H * self.scale, : orig_W * self.scale]


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
