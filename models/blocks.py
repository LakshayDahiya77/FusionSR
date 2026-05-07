import math
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


# ─────────────────────────────────────────
#  Swin Transformer components
# ─────────────────────────────────────────


def window_partition(x: torch.Tensor, window_size: int):
    """
    Split feature map into non-overlapping windows.
    x: [B, H, W, C]
    returns: [num_windows*B, window_size, window_size, C]

    """

    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int):
    """
    Reconstruct feature map from windows.
    windows: [num_windows*B, window_size, window_size, C]
    returns: [B, H, W, C]

    """

    B_times_nW = windows.shape[0]
    nW = (H // window_size) * (W // window_size)
    B = B_times_nW // nW
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention (W-MSA / SW-MSA).
    Includes relative position bias.

    """

    def __init__(self, channels: int, window_size: int, num_heads: int):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(channels, channels * 3, bias=True)
        self.proj = nn.Linear(channels, channels, bias=True)

        # relative position bias table
        self.rel_pos_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)

        # precompute relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij")
        )  # [2, W, W]
        coords_flat = coords.flatten(1)  # [2, W*W]
        relative = coords_flat[:, :, None] - coords_flat[:, None, :]  # [2, W*W, W*W]
        relative = relative.permute(1, 2, 0).contiguous()  # [W*W, W*W, 2]
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        rel_pos_index = relative.sum(-1)  # [W*W, W*W]
        self.register_buffer("rel_pos_index", rel_pos_index)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B_, N, C = x.shape  # B_ = num_windows * B, N = window_size^2
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each [B_, num_heads, N, head_dim]

        attn = (q * self.scale) @ k.transpose(-2, -1)  # [B_, heads, N, N]

        # add relative position bias
        bias = self.rel_pos_bias_table[self.rel_pos_index.view(-1)]
        bias = bias.view(self.window_size**2, self.window_size**2, self.num_heads)
        bias = bias.permute(2, 0, 1).contiguous().unsqueeze(0)  # [1, heads, N, N]
        attn = attn + bias

        # apply shift mask if SW-MSA
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinBlock(nn.Module):
    """
    One Swin Transformer block — either W-MSA or SW-MSA.
    LayerNorm → attention → skip
    LayerNorm → FFN → skip

    """

    def __init__(
        self,
        channels: int,
        window_size: int,
        num_heads: int,
        shift: bool = False,  # False = W-MSA, True = SW-MSA
        ffn_ratio: float = 2.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0

        self.norm1 = nn.LayerNorm(channels)
        self.attn = WindowAttention(channels, window_size, num_heads)
        self.norm2 = nn.LayerNorm(channels)

        ffn_dim = int(channels * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, channels),
        )

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C] — Swin works in HWC

        shortcut = x
        x = self.norm1(x)

        # cyclic shift for SW-MSA
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # partition into windows
        x_windows = window_partition(x, self.window_size)  # [nW*B, ws, ws, C]
        x_windows = x_windows.view(-1, self.window_size**2, C)  # [nW*B, ws^2, C]

        # attention
        x_windows = self.attn(x_windows, mask=attn_mask)

        # reverse windows
        x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x_windows, self.window_size, H, W)  # [B, H, W, C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + x

        # FFN
        x = x + self.ffn(self.norm2(x))

        x = x.permute(0, 3, 1, 2)  # back to [B, C, H, W]
        return x


class SwinBlockPair(nn.Module):
    """
    W-MSA block followed by SW-MSA block.
    The pair ensures cross-boundary information flow.
    Precomputes the shift mask once for efficiency.

    """

    def __init__(self, channels: int, window_size: int, num_heads: int):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2

        self.w_msa = SwinBlock(channels, window_size, num_heads, shift=False)
        self.sw_msa = SwinBlock(channels, window_size, num_heads, shift=True)

        self._attn_mask = None  # computed lazily on first forward

    def _compute_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
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

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, ws, ws, 1]
        mask_windows = mask_windows.view(-1, self.window_size**2)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # compute mask once, reuse if spatial dims unchanged
        if self._attn_mask is None or self._attn_mask.device != x.device:
            self._attn_mask = self._compute_mask(H, W, x.device)

        x = self.w_msa(x, attn_mask=None)
        x = self.sw_msa(x, attn_mask=self._attn_mask)
        return x


# ─────────────────────────────────────────
#  Dual-Path Shallow Feature Extractor
# ─────────────────────────────────────────
class DualPathExtractor(nn.Module):
    """
    Two parallel paths for shallow feature extraction (from FusionSR).

    Path A — general: simple conv stack, fast, low-level features
    Path B — attention: same but with two RCAB blocks for richer features

    Output is a learned weighted blend:
        out = (1 - sigmoid(w)) * A + sigmoid(w) * B
        w initialized to -2.0 → sigmoid ≈ 0.12 (mostly Path A early in training)

    The network learns how much to rely on the attention path.

    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Path A — general
        self.path_a = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=True),
        )

        # Path B — attention enhanced
        self.path_b = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
            nn.GELU(),
            RCAB(out_channels),
            RCAB(out_channels),
            nn.Conv2d(out_channels, out_channels, 1, bias=True),
        )

        # learnable mixing scalar — init at -2.0 so sigmoid ≈ 0.12
        self.mix_weight = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.path_a(x)
        b = self.path_b(x)
        mix = torch.sigmoid(self.mix_weight)
        return (1.0 - mix) * a + mix * b


# ─────────────────────────────────────────
#  Residual Group
# ─────────────────────────────────────────
class ResidualGroup(nn.Module):
    """
    One residual group — the repeating unit of Stage 2.

    Structure:
        4x RCAB blocks  (local features, channel attention)
        1x SwinBlockPair (global context, shifted window attention)
        Conv 3x3
        Group-level skip connection

    The group-level skip means the entire group only needs to
    learn a residual, not a full transformation.

    """

    def __init__(
        self,
        channels: int,
        window_size: int,
        num_heads: int,
        num_rcab: int = 4,
    ):
        super().__init__()

        self.rcab_blocks = nn.Sequential(*[RCAB(channels) for _ in range(num_rcab)])
        self.swin_pair = SwinBlockPair(channels, window_size, num_heads)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.rcab_blocks(x)
        res = self.swin_pair(res)
        res = self.conv(res)
        return x + res  # group skip
