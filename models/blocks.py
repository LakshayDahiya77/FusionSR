import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────
#  Channel Attention
# ─────────────────────────────────────────
class ChannelAttention(nn.Module):
    """
    Squeeze-and-excite channel attention from RCAN.
    1. GAP: squeezes spatial dims → [B, C, 1, 1]
    2. Two FC layers learn channel importance scores
    3. Sigmoid gates each channel in [0, 1]
    4. Scale original feature map by these gates

    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.gap(x))  # [B, C, 1, 1]
        return x * scale  # broadcast across H, W


# ─────────────────────────────────────────
#  RCAB — Residual Channel Attention Block
# ─────────────────────────────────────────
class RCAB(nn.Module):
    """
    Residual Channel Attention Block from RCAN.
    conv → GELU → conv → ChannelAttention → residual scaling → skip
    No BatchNorm anywhere (EDSR principle).

    """

    def __init__(self, channels: int, reduction: int = 16, res_scale: float = 0.1):
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
        )
        self.ca = ChannelAttention(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.ca(self.body(x)) * self.res_scale
        return x + res
