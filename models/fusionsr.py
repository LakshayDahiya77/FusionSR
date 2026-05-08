import torch
import torch.nn as nn
from models.blocks import DualPathExtractor, ResidualGroup


class FusionSR(nn.Module):
    """
    FusionSR — Hybrid CNN-Transformer for 4x Satellite Image Super Resolution.

    Stage 1 — Dual-path shallow extractor (FusionSR original)
        Learnable mixing between general conv path and attention-enhanced path.

    Stage 2 — Hybrid deep feature extraction body
        N residual groups, each containing:
            4x RCAB blocks  (local features + channel attention)
            1x Swin pair    (global context via shifted window attention)
            Conv 3x3 + group skip connection
        Long skip connection over entire body.

    Stage 3 — Reconstruction
        Conv → PixelShuffle (4x) → cleanup conv

    """

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
    ):
        super().__init__()
        self.scale = scale

        # ── Stage 1 — shallow extractor ──
        self.shallow = DualPathExtractor(in_channels, channels)

        # ── Stage 2 — deep feature extraction ──
        self.body = nn.Sequential(
            *[
                ResidualGroup(channels, window_size, num_heads, num_rcab)
                for _ in range(num_groups)
            ]
        )
        self.body_conv = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

        # ── Stage 3 — reconstruction ──
        self.reconstruction = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.Conv2d(channels, out_channels * scale * scale, 3, padding=1, bias=True),
            nn.PixelShuffle(scale),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1
        shallow = self.shallow(x)  # [B, C, H, W]

        # Stage 2
        deep = self.body(shallow)  # [B, C, H, W]
        deep = self.body_conv(deep)

        # long skip — add shallow features to deep output
        fused = deep + shallow  # [B, C, H, W]

        # Stage 3
        out = self.reconstruction(fused)  # [B, out_ch, 4H, 4W]
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
