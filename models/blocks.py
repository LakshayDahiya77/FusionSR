"""
FusionSR-v5 building blocks.

Novel components (with paper origins and modifications):

    HighFreqEnhancementBranch — Inspired by CRAFT (Li et al., ICCV 2023).
        CRAFT uses parallel HFERB + SRWAB + HFB blocks.  We simplify to a
        lightweight depthwise-pointwise conv that extracts the high-frequency
        residual and re-weights it with a learnable per-channel scale.
        Replaces v4's 6-RCAB stack at 58× fewer parameters.

    HybridSwinBlock — Inspired by HAT (Chen et al., CVPR 2023).
        HAT fuses channel attention with window self-attention inside each
        Hybrid Attention Block.  We integrate a squeeze-excite channel gate
        on the attention residual *before* the skip addition, giving each
        block a unified spatial-channel attention mechanism.  SwinIR has no
        channel attention at all; v4 placed it as a separate CAB stage.

    TokenDictionaryCrossAttention — Inspired by ATD (Li et al., CVPR 2024).
        ATD uses a learnable token dictionary as the *entire* attention
        mechanism.  We use it as a *complement* to windowed self-attention:
        image features cross-attend to a shared dictionary for global
        self-similarity with O(N×K) cost.  Inserted every 2nd group.

    MultiScaleWindowGroup — Novel combination (not in any published SR model).
        All existing windowed-attention SR models (SwinIR, HAT, DRCT) use a
        single fixed window size.  We alternate ws=4 (fine local edges) and
        ws=8 (medium-range patterns) within each group, giving multi-scale
        receptive fields at no additional parameter cost.

Retained from v4:
    ChannelLayerNorm  — LayerNorm for BCHW tensors
    GDFN              — Gated-DConv FFN from Restormer (Zamir et al., 2022)
    WindowAttention   — Window multi-head self-attention (SwinIR)
    window_partition / window_reverse — Standard Swin utilities

Removed from v4:
    RCAB, ChannelAttention, ChannelAttentionBridge, OverlappingCrossAttention,
    SwinBlock (replaced by HybridSwinBlock), SwinBlockPair (replaced by MSWA)
"""

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
#  High-Frequency Enhancement Branch (HFEB)
# ─────────────────────────────────────────

class HighFreqEnhancementBranch(nn.Module):
    """
    Explicit high-frequency feature enhancement — replaces v4's RCAB stack.

    Source: Inspired by CRAFT (Li et al., ICCV 2023) which demonstrated that
    transformers have a low-frequency bias and introduced HFERB blocks for
    explicit high-frequency extraction.

    Difference from CRAFT: CRAFT uses parallel HFERB + SRWAB + HFB (three
    separate blocks with cross-refinement).  We simplify to a lightweight
    depthwise-pointwise convolution that extracts the high-frequency residual
    (local features minus input = HF component) and re-weights it with a
    learnable per-channel scale.  This acts as a preprocessing step before
    the transformer layers, not a parallel branch.

    Cost: ~60K params per instance (vs ~3.5M for 6 RCABs in v4).

    When disabled (use_hfeb=False): acts as identity passthrough.
    """

    def __init__(self, channels: int, hf_scale_init: float = 0.01):
        super().__init__()
        # Depthwise 3×3 for spatial edge extraction (very cheap)
        self.dw_conv = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=True
        )
        # 1×1 pointwise to mix channels
        self.pw_conv = nn.Conv2d(channels, channels, 1, bias=True)
        self.act = nn.GELU()
        # Learnable per-channel HF emphasis scale
        # Initialized small (0.01) to prevent instability in early training
        self.hf_scale = nn.Parameter(
            torch.ones(1, channels, 1, 1) * hf_scale_init
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract local features via depthwise + pointwise
        local = self.act(self.pw_conv(self.dw_conv(x)))
        # The residual IS the high-frequency component
        hf = local - x
        # Amplify HF with learnable per-channel scale
        return x + hf * self.hf_scale


# ─────────────────────────────────────────
#  Swin Transformer Utilities
# ─────────────────────────────────────────

def window_partition(x: torch.Tensor, window_size: int):
    """Split [B, H, W, C] feature map into non-overlapping windows.
    Returns [num_windows*B, window_size, window_size, C].
    """
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int):
    """Reconstruct [B, H, W, C] feature map from windows.
    windows: [num_windows*B, window_size, window_size, C].
    """
    nW = (H // window_size) * (W // window_size)
    B = windows.shape[0] // nW
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H, W, -1)


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
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)  # each [B_, num_heads, N, head_dim]

        # relative position bias: [1, num_heads, N, N]
        bias = self.rel_pos_bias_table[self.rel_pos_index.view(-1)]
        bias = bias.reshape(N, N, self.num_heads).permute(2, 0, 1).unsqueeze(0).contiguous()

        # combine bias with shift mask for fused SDPA kernel
        if mask is not None:
            nW = mask.shape[0]
            B = B_ // nW
            # mask [nW, N, N] → tile across batch → [B*nW, 1, N, N]
            attn_mask = (bias + mask.repeat(B, 1, 1).unsqueeze(1)).contiguous()
        else:
            attn_mask = bias
        
        # ensure mask matches query dtype (crucial for AMP float16)
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=q.dtype)

        # fused attention: Q·K^T scaling + mask + softmax + V in one kernel
        with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, scale=self.scale
            )
        x = x.transpose(1, 2).contiguous().reshape(B_, N, C)
        return self.proj(x)


# ─────────────────────────────────────────
#  HybridSwinBlock — W-MSA + Channel Attention Gate
# ─────────────────────────────────────────

class HybridSwinBlock(nn.Module):
    """
    Swin Transformer block with integrated channel attention gating.

    Source: Channel attention fusion from HAT (Chen et al., CVPR 2023).
    HAT combines channel attention with window self-attention in its Hybrid
    Attention Block.  SwinIR's standard SwinBlock has NO channel attention.

    Difference from HAT: HAT uses a separate channel attention layer followed
    by window attention within the HAB.  We apply a squeeze-excite channel
    gate directly on the attention output *before* the residual addition.
    This is a multiplicative gate, not a separate layer — the channel
    attention modulates which spatial attention channels to keep.

    Difference from v4: v4 placed channel attention as a separate CAB stage
    BETWEEN the RCAB stack and SwinBlockPair.  Here it is INSIDE each
    transformer block, giving per-block spatial-channel fusion.

    When disabled (use_hybrid_ca=False): falls back to standard SwinBlock
    residual (no channel gating).

    Structure:
        Attention branch: LayerNorm → [shift] → window partition → attention →
                          window reverse → [unshift] → channel_gate → skip
        FFN branch:       ChannelLayerNorm → GDFN → skip
    """

    def __init__(
        self,
        channels: int,
        window_size: int,
        num_heads: int,
        shift: bool = False,
        ffn_expansion: float = 2.0,
        use_hybrid_ca: bool = True,
        ca_reduction: int = 16,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0
        self.use_hybrid_ca = use_hybrid_ca

        # attention branch (BHWC)
        self.norm1 = nn.LayerNorm(channels)
        self.attn = WindowAttention(channels, window_size, num_heads)

        # channel attention gate on attention output (HAT-inspired)
        if use_hybrid_ca:
            ca_mid = max(channels // ca_reduction, 4)
            self.channel_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, ca_mid, 1, bias=True),
                nn.GELU(),
                nn.Conv2d(ca_mid, channels, 1, bias=True),
                nn.Sigmoid(),
            )

        # GDFN branch (BCHW)
        self.norm2 = ChannelLayerNorm(channels)
        self.gdfn = GDFN(channels, expansion=ffn_expansion)

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        B, C, H, W = x.shape

        # ── attention branch (BHWC) ──
        x_bhwc = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x_bhwc = self.norm1(x_bhwc)

        # cyclic shift for SW-MSA
        if self.shift_size > 0:
            x_bhwc = x_bhwc.contiguous()
            x_bhwc = torch.roll(
                x_bhwc, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            ).contiguous()

        # window partition → attention → window reverse
        windows = window_partition(x_bhwc, self.window_size)
        windows = windows.reshape(-1, self.window_size ** 2, C)
        windows = self.attn(windows, mask=attn_mask)
        windows = windows.reshape(-1, self.window_size, self.window_size, C)
        x_bhwc = window_reverse(windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x_bhwc = x_bhwc.contiguous()
            x_bhwc = torch.roll(
                x_bhwc, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            ).contiguous()

        # Convert to BCHW for channel gate
        attn_out = x_bhwc.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        # Channel attention gate (HAT-style hybrid attention)
        if self.use_hybrid_ca:
            gate = self.channel_gate(attn_out)  # [B, C, 1, 1]
            attn_out = (attn_out * gate).contiguous()

        # attention skip (in BCHW)
        x = x + attn_out

        # ── GDFN branch (BCHW) ──
        x = x + self.gdfn(self.norm2(x))

        return x


# ─────────────────────────────────────────
#  Multi-Scale Window Group (MSWA)
# ─────────────────────────────────────────

class MultiScaleWindowGroup(nn.Module):
    """
    4 HybridSwinBlocks with alternating window sizes for multi-scale attention.

    Source: Novel for SR.  Multi-scale window attention exists in detection
    (Swin v2, PVT) but NO published SR model (SwinIR, HAT, DRCT) uses
    mixed window sizes within a single residual group.

    Layout:
        Block 0: W-MSA  at ws=4  (fine-grained local edges, 16 tokens)
        Block 1: SW-MSA at ws=4  (fine shifted, cross-boundary at 4px scale)
        Block 2: W-MSA  at ws=8  (medium-range context, 64 tokens)
        Block 3: SW-MSA at ws=8  (medium shifted, cross-boundary at 8px scale)

    When disabled (use_mswa=False): all 4 blocks use uniform ws=8.

    Precomputes shift masks for both window sizes.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        ffn_expansion: float = 2.0,
        use_mswa: bool = True,
        use_hybrid_ca: bool = True,
    ):
        super().__init__()
        self.use_mswa = use_mswa

        ws_fine = 4 if use_mswa else 8
        ws_med = 8

        self.blocks = nn.ModuleList([
            HybridSwinBlock(
                channels, ws_fine, num_heads,
                shift=False, ffn_expansion=ffn_expansion,
                use_hybrid_ca=use_hybrid_ca,
            ),
            HybridSwinBlock(
                channels, ws_fine, num_heads,
                shift=True, ffn_expansion=ffn_expansion,
                use_hybrid_ca=use_hybrid_ca,
            ),
            HybridSwinBlock(
                channels, ws_med, num_heads,
                shift=False, ffn_expansion=ffn_expansion,
                use_hybrid_ca=use_hybrid_ca,
            ),
            HybridSwinBlock(
                channels, ws_med, num_heads,
                shift=True, ffn_expansion=ffn_expansion,
                use_hybrid_ca=use_hybrid_ca,
            ),
        ])

        # Cache shift masks per window size
        self._masks = {}

    def _compute_mask(
        self, H: int, W: int, window_size: int, device: torch.device
    ) -> torch.Tensor:
        """Compute attention mask for shifted window self-attention."""
        shift_size = window_size // 2
        img_mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        w_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, window_size)
        mask_windows = mask_windows.reshape(-1, window_size ** 2)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        return attn_mask.contiguous()

    def _get_mask(
        self, H: int, W: int, window_size: int, device: torch.device
    ) -> torch.Tensor:
        """Get or compute cached shift mask for a given spatial size + window size."""
        key = (H, W, window_size)
        if key not in self._masks:
            self._masks[key] = self._compute_mask(H, W, window_size, device)
        return self._masks[key].to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape

        for block in self.blocks:
            if block.shift_size > 0:
                mask = self._get_mask(H, W, block.window_size, x.device)
            else:
                mask = None
            x = block(x, attn_mask=mask)

        return x


# ─────────────────────────────────────────
#  Token Dictionary Cross-Attention (TDCA)
# ─────────────────────────────────────────

class TokenDictionaryCrossAttention(nn.Module):
    """
    Learnable token dictionary for global self-similarity matching.

    Source: ATD (Li et al., CVPR 2024) — Adaptive Token Dictionary.
    ATD uses a learnable dictionary as the *entire* attention mechanism,
    replacing window-based self-attention.

    Difference from ATD: We use TDCA as a *complement* to windowed
    self-attention, not a replacement.  Image features cross-attend to a
    shared dictionary for global pattern matching, while W-MSA/SW-MSA
    handles local spatial context.  This combination (windowed SA + TDCA)
    is novel — no published model uses both.

    Complexity: O(N × K × C) where K=num_tokens is fixed (typically 64),
    making this LINEAR in spatial size N.  Compare to O(N²) for full
    self-attention or O(N × W²) for windowed attention.

    When disabled (use_tdca=False): skipped entirely (identity).

    Implementation note: The dictionary [K, C] is explicitly expanded to
    [B, K, C] before the attention matmul — no implicit broadcasting.
    """

    def __init__(
        self,
        channels: int,
        num_tokens: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.channels = channels
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        # Learnable token dictionary [K, C]
        self.dictionary = nn.Parameter(torch.randn(num_tokens, channels) * 0.02)

        self.norm = nn.LayerNorm(channels)
        self.q_proj = nn.Linear(channels, channels, bias=True)
        self.k_proj = nn.Linear(channels, channels, bias=True)
        self.v_proj = nn.Linear(channels, channels, bias=True)
        self.out_proj = nn.Linear(channels, channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        # Flatten to [B, N, C] for attention
        x_flat = x.permute(0, 2, 3, 1).reshape(B, N, C)
        shortcut = x_flat

        x_norm = self.norm(x_flat)

        # Q from image features: [B, N, C]
        Q = self.q_proj(x_norm)

        # K, V from dictionary: [K, C] → explicitly expand to [B, K, C]
        dict_expanded = self.dictionary.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, K, C]
        K = self.k_proj(dict_expanded)  # [B, K, C]
        V = self.v_proj(dict_expanded)  # [B, K, C]

        # Reshape for multi-head attention
        Q = Q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        K = K.reshape(B, self.num_tokens, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        V = V.reshape(B, self.num_tokens, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        # Q: [B, H, N, D], K: [B, H, K, D], V: [B, H, K, D]

        # Fused scaled dot-product cross-attention
        with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
            out = F.scaled_dot_product_attention(Q, K, V, scale=self.scale)
        # out: [B, H, N, D]
        out = out.transpose(1, 2).contiguous().reshape(B, N, C)
        out = self.out_proj(out)

        # Residual connection and reshape back to BCHW
        out = shortcut + out
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)


# ─────────────────────────────────────────
#  Residual Group (v5)
# ─────────────────────────────────────────

class ResidualGroup(nn.Module):
    """
    Residual group — the repeating unit of FusionSR-v5's deep feature extraction.

    Structure:
        HFEB                     (explicit HF feature emphasis — CRAFT-inspired)
        MultiScaleWindowGroup    (4 HybridSwinBlocks at ws=4 and ws=8)
        Conv 3×3                 (feature refinement)
        Group-level skip         (residual learning)

    Changes from v4:
        - RCAB×6 stack → HFEB (58× fewer params, targeted HF extraction)
        - CAB → removed (channel attention now inside each HybridSwinBlock)
        - SwinBlockPair → MultiScaleWindowGroup (mixed ws=4+ws=8)
        - OCA → removed (TDCA handles global context at model level, not group level)
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        ffn_expansion: float = 2.0,
        use_hfeb: bool = True,
        use_mswa: bool = True,
        use_hybrid_ca: bool = True,
        hf_scale_init: float = 0.01,
    ):
        super().__init__()
        self.use_hfeb = use_hfeb

        # High-frequency enhancement (replaces RCAB stack)
        if use_hfeb:
            self.hfeb = HighFreqEnhancementBranch(channels, hf_scale_init)

        # Multi-scale window attention (4 HybridSwinBlocks)
        self.mswa = MultiScaleWindowGroup(
            channels, num_heads, ffn_expansion,
            use_mswa=use_mswa,
            use_hybrid_ca=use_hybrid_ca,
        )

        # Refinement conv
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        if self.use_hfeb:
            res = self.hfeb(res)        # HF emphasis
        res = self.mswa(res)            # multi-scale windowed attention
        res = self.conv(res)            # refinement
        return x + res                  # group skip
