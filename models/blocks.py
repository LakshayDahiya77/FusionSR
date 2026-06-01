"""
FusionSR-v3 building blocks.

Components (from literature):
    ChannelAttention       — squeeze-excite from RCAN (Zhang et al., 2018)
    RCAB                   — residual channel attention block (RCAN)
    GDFN                   — gated-dconv FFN from Restormer (Zamir et al., 2022)
    ChannelAttentionBridge — cross-window bridge inspired by HAT (Chen et al., 2023)
    WindowAttention        — window multi-head self-attention (SwinIR, Liang et al., 2021)
    SwinBlock              — W-MSA / SW-MSA + GDFN (v3 upgrade)
    SwinBlockPair          — W-MSA then SW-MSA with precomputed shift mask
    ResidualGroup          — RCAB×N → CAB → SwinBlockPair → Conv → skip

Changes from v2:
    - Standard MLP FFN replaced with GDFN in SwinBlock
    - ChannelAttentionBridge inserted between RCAB and Swin stages
    - DualPathExtractor removed (replaced by single conv in fusionsr.py)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────
#  Normalization Utilities
# ─────────────────────────────────────────

class ChannelLayerNorm(nn.Module):
    """
    LayerNorm for [B, C, H, W] tensors.
    Internally permutes to [B, H, W, C], applies LayerNorm, permutes back.
    Used before GDFN which operates in BCHW format.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# ─────────────────────────────────────────
#  Channel Attention (RCAN)
# ─────────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    Squeeze-and-excite channel attention from RCAN.
    GAP → FC → ReLU → FC → Sigmoid → scale.
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
        return x * self.fc(self.gap(x))


# ─────────────────────────────────────────
#  RCAB — Residual Channel Attention Block
# ─────────────────────────────────────────

class RCAB(nn.Module):
    """
    Residual Channel Attention Block from RCAN.
    Conv → GELU → Conv → ChannelAttention → residual scaling → skip.
    No BatchNorm (EDSR principle).
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
        return x + self.ca(self.body(x)) * self.res_scale


# ─────────────────────────────────────────
#  GDFN — Gated-DConv Feed-Forward Network
# ─────────────────────────────────────────

class GDFN(nn.Module):
    """
    Gated-DConv Feed-Forward Network from Restormer (Zamir et al., 2022 §3.2).

    Replaces standard Linear→GELU→Linear FFN with a spatially-aware
    gated network:
        1. 1×1 conv projects to 2×hidden channels (for gating)
        2. Depthwise 3×3 conv injects local spatial context
        3. Split into two halves — one gates the other via GELU
        4. 1×1 conv projects back to original channels

    This addresses v2's limitation of spatial-unaware FFN in Swin blocks.
    Operates in [B, C, H, W] format (Conv2d-based).
    """

    def __init__(self, channels: int, expansion: float = 2.0):
        super().__init__()
        hidden = int(channels * expansion)
        # expand to 2×hidden for the gating split
        self.project_in = nn.Conv2d(channels, hidden * 2, 1, bias=True)
        # depthwise conv adds 3×3 local spatial context
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=True
        )
        self.project_out = nn.Conv2d(hidden, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)         # [B, 2*hidden, H, W]
        x = self.dwconv(x)             # depthwise 3×3 for spatial awareness
        x1, x2 = x.chunk(2, dim=1)     # each [B, hidden, H, W]
        x = x1 * F.gelu(x2)            # gating: x1 modulated by activated x2
        return self.project_out(x)      # [B, C, H, W]


# ─────────────────────────────────────────
#  Channel Attention Bridge (HAT-inspired)
# ─────────────────────────────────────────

class ChannelAttentionBridge(nn.Module):
    """
    Channel attention bridge inspired by HAT (Chen et al., 2023).

    Placed between the RCAB stack (local CNN features) and SwinBlockPair
    (windowed transformer attention). The global average pooling aggregates
    information across ALL spatial positions, enabling cross-window
    information flow before the windowed self-attention stage.

    This bridges the gap: RCAB produces local features confined to conv
    receptive fields → CAB creates a globally-informed representation →
    Swin blocks attend within windows but start from globally-aware features.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(mid, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.body(x)  # global channel gating


# ─────────────────────────────────────────
#  Swin Transformer Components
# ─────────────────────────────────────────

def window_partition(x: torch.Tensor, window_size: int):
    """Split [B, H, W, C] feature map into non-overlapping windows.
    Returns [num_windows*B, window_size, window_size, C].
    """
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int):
    """Reconstruct [B, H, W, C] feature map from windows.
    windows: [num_windows*B, window_size, window_size, C].
    """
    nW = (H // window_size) * (W // window_size)
    B = windows.shape[0] // nW
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with relative position bias.
    Used for both W-MSA and SW-MSA (SwinIR, Liang et al., 2021).
    """

    def __init__(self, channels: int, window_size: int, num_heads: int):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(channels, channels * 3, bias=True)
        self.proj = nn.Linear(channels, channels, bias=True)

        # relative position bias table and index
        self.rel_pos_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flat = coords.flatten(1)
        relative = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("rel_pos_index", relative.sum(-1))

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each [B_, num_heads, N, head_dim]

        # relative position bias: [1, num_heads, N, N]
        bias = self.rel_pos_bias_table[self.rel_pos_index.view(-1)]
        bias = bias.reshape(N, N, self.num_heads).permute(2, 0, 1).unsqueeze(0)

        # combine bias with shift mask for fused SDPA kernel
        if mask is not None:
            nW = mask.shape[0]
            B = B_ // nW
            # mask [nW, N, N] → tile across batch → [B*nW, 1, N, N]
            attn_mask = bias + mask.repeat(B, 1, 1).unsqueeze(1)
        else:
            attn_mask = bias

        # fused attention: Q·K^T scaling + mask + softmax + V in one kernel
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=self.scale
        )
        x = x.transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinBlock(nn.Module):
    """
    Swin Transformer block with GDFN (v3 upgrade).

    Attention branch: LayerNorm → [shift] → window partition → attention →
                      window reverse → [unshift] → skip
    FFN branch:       ChannelLayerNorm → GDFN → skip

    Key change from v2: standard Linear→GELU→Linear FFN replaced with GDFN
    for local spatial awareness in the feed-forward path.

    Attention operates in [B, H, W, C]; GDFN operates in [B, C, H, W].
    """

    def __init__(
        self,
        channels: int,
        window_size: int,
        num_heads: int,
        shift: bool = False,
        ffn_expansion: float = 2.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0

        # attention branch (BHWC)
        self.norm1 = nn.LayerNorm(channels)
        self.attn = WindowAttention(channels, window_size, num_heads)

        # GDFN branch (BCHW)
        self.norm2 = ChannelLayerNorm(channels)
        self.gdfn = GDFN(channels, expansion=ffn_expansion)

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        B, C, H, W = x.shape

        # ── attention branch (BHWC) ──
        x_bhwc = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        shortcut = x_bhwc
        x_bhwc = self.norm1(x_bhwc)

        # cyclic shift for SW-MSA
        if self.shift_size > 0:
            x_bhwc = torch.roll(
                x_bhwc, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )

        # window partition → attention → window reverse
        windows = window_partition(x_bhwc, self.window_size)
        windows = windows.reshape(-1, self.window_size ** 2, C)
        windows = self.attn(windows, mask=attn_mask)
        windows = windows.reshape(-1, self.window_size, self.window_size, C)
        x_bhwc = window_reverse(windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x_bhwc = torch.roll(
                x_bhwc, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )

        x_bhwc = shortcut + x_bhwc  # attention skip

        # ── GDFN branch (BCHW) ──
        x = x_bhwc.permute(0, 3, 1, 2)   # [B, C, H, W]
        x = x + self.gdfn(self.norm2(x))  # GDFN skip

        return x


class SwinBlockPair(nn.Module):
    """
    W-MSA block followed by SW-MSA block.
    The pair ensures cross-boundary information flow via shifted windows.
    Precomputes the shift mask once for efficiency.
    """

    def __init__(
        self,
        channels: int,
        window_size: int,
        num_heads: int,
        ffn_expansion: float = 2.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2

        self.w_msa = SwinBlock(
            channels, window_size, num_heads, shift=False, ffn_expansion=ffn_expansion
        )
        self.sw_msa = SwinBlock(
            channels, window_size, num_heads, shift=True, ffn_expansion=ffn_expansion
        )
        self._attn_mask = None

    def _compute_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Compute attention mask for shifted window self-attention."""
        img_mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.reshape(-1, self.window_size ** 2)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # recompute mask if spatial size or device changed
        if (
            self._attn_mask is None
            or self._attn_mask.device != x.device
            or self._attn_mask.shape[0]
            != (H // self.window_size) * (W // self.window_size)
        ):
            self._attn_mask = self._compute_mask(H, W, x.device)

        x = self.w_msa(x, attn_mask=None)
        x = self.sw_msa(x, attn_mask=self._attn_mask)
        return x


# ─────────────────────────────────────────
#  Residual Group
# ─────────────────────────────────────────

class ResidualGroup(nn.Module):
    """
    Residual group — the repeating unit of Stage 2 (v3).

    Structure:
        RCAB × N                (local features, channel attention)
        ChannelAttentionBridge  (global channel bridge — HAT-inspired)
        SwinBlockPair           (global spatial context, shifted window + GDFN)
        Conv 3×3                (feature refinement)
        Group-level skip        (residual learning)

    The CAB between RCAB and Swin enables cross-window information flow:
    RCAB outputs are locally confined → CAB applies global channel gating →
    Swin blocks start from globally-informed features.
    """

    def __init__(
        self,
        channels: int,
        window_size: int,
        num_heads: int,
        num_rcab: int = 6,
        ffn_expansion: float = 2.0,
    ):
        super().__init__()
        self.rcab_blocks = nn.Sequential(*[RCAB(channels) for _ in range(num_rcab)])
        self.cab = ChannelAttentionBridge(channels)
        self.swin_pair = SwinBlockPair(channels, window_size, num_heads, ffn_expansion)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.rcab_blocks(x)   # local CNN features
        res = self.cab(res)         # global channel bridge
        res = self.swin_pair(res)   # global spatial attention + GDFN
        res = self.conv(res)        # refinement
        return x + res              # group skip
