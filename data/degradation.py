"""
FusionSR-v3 Real-ESRGAN second-order degradation pipeline (GPU-based).

All operations run entirely on CUDA tensors — no CPU↔GPU transfers.
This models realistic image degradations beyond simple bicubic downscaling:
    blur → resize → noise → JPEG  (first degradation)
    blur → resize → noise → JPEG  (second degradation)

Reference: Wang et al., "Real-ESRGAN: Training Real-World Blind
Super-Resolution with Pure Synthetic Data" (2021), §3.2.

Only activated when CONFIG['use_degradation'] = True. Disabled by default.
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────
#  Gaussian Blur
# ─────────────────────────────────────────

def generate_gaussian_kernel(kernel_size: int, sigma: float, device: torch.device):
    """Generate 2D isotropic Gaussian kernel on GPU.
    Returns [1, 1, kernel_size, kernel_size] tensor, normalized to sum=1.
    """
    x = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
    gauss_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gauss_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)
    gauss_2d = gauss_2d / gauss_2d.sum()
    return gauss_2d.unsqueeze(0).unsqueeze(0)


def apply_blur(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply blur kernel to image batch via depthwise convolution.
    img: [B, C, H, W], kernel: [1, 1, k, k].
    """
    C = img.shape[1]
    pad = kernel.shape[-1] // 2
    kernel_expanded = kernel.expand(C, -1, -1, -1)  # [C, 1, k, k]
    return F.conv2d(F.pad(img, [pad] * 4, mode="reflect"), kernel_expanded, groups=C)


def random_blur(img: torch.Tensor, kernel_sizes: list, sigma_range: tuple):
    """Apply random Gaussian blur."""
    kernel_size = random.choice(kernel_sizes)
    sigma = random.uniform(*sigma_range)
    kernel = generate_gaussian_kernel(kernel_size, sigma, img.device)
    return apply_blur(img, kernel)


# ─────────────────────────────────────────
#  Random Resize
# ─────────────────────────────────────────

def random_resize(
    img: torch.Tensor,
    scale_range: tuple,
    modes: list = ("bilinear", "bicubic", "area"),
) -> torch.Tensor:
    """Randomly resize image within scale range using random interpolation mode."""
    scale = random.uniform(*scale_range)
    mode = random.choice(modes)
    H, W = img.shape[2], img.shape[3]
    new_H = max(int(H * scale), 1)
    new_W = max(int(W * scale), 1)

    if mode == "area":
        return F.interpolate(img, size=(new_H, new_W), mode="area")
    else:
        return F.interpolate(
            img, size=(new_H, new_W), mode=mode, align_corners=False, antialias=True
        )


# ─────────────────────────────────────────
#  Noise
# ─────────────────────────────────────────

def add_gaussian_noise(img: torch.Tensor, sigma_range: tuple) -> torch.Tensor:
    """Add Gaussian noise with random sigma.
    sigma_range: (min, max) in [0, 255] scale.
    """
    sigma = random.uniform(*sigma_range) / 255.0
    return img + torch.randn_like(img) * sigma


def add_poisson_noise(img: torch.Tensor, scale_range: tuple) -> torch.Tensor:
    """Add Poisson-like noise using Gaussian approximation.
    scale_range controls noise intensity.
    """
    scale = random.uniform(*scale_range)
    if scale < 1e-6:
        return img
    # Poisson noise variance = signal level / scale
    noise = torch.randn_like(img) * torch.sqrt(img.clamp(min=1e-6) / scale)
    return img + noise


# ─────────────────────────────────────────
#  DiffJPEG — GPU-based JPEG Simulation
# ─────────────────────────────────────────

class DiffJPEG(nn.Module):
    """
    GPU-based JPEG compression simulation using block DCT.

    Pipeline: RGB → YCbCr → 8×8 block DCT → quantize → dequantize → IDCT → RGB

    Uses standard JPEG quantization tables scaled by quality factor.
    Operates entirely on GPU tensors via matrix multiplication for DCT.
    """

    def __init__(self):
        super().__init__()

        # precompute 8×8 DCT-II matrix
        dct = self._create_dct_matrix(8)
        self.register_buffer("dct_mat", dct)          # [8, 8]
        self.register_buffer("idct_mat", dct.t())      # [8, 8] — DCT-III = DCT-II^T

        # standard JPEG luminance quantization table
        luma = torch.tensor(
            [
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("luma_table", luma)

        # standard JPEG chrominance quantization table
        chroma = torch.tensor(
            [
                [17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("chroma_table", chroma)

    @staticmethod
    def _create_dct_matrix(n: int) -> torch.Tensor:
        """Create n×n DCT-II transform matrix."""
        dct = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                if i == 0:
                    dct[i, j] = 1.0 / math.sqrt(n)
                else:
                    dct[i, j] = math.sqrt(2.0 / n) * math.cos(
                        (2 * j + 1) * i * math.pi / (2 * n)
                    )
        return dct

    def _scale_table(self, table: torch.Tensor, quality: int) -> torch.Tensor:
        """Scale quantization table by JPEG quality factor (1-100)."""
        if quality < 50:
            scale = 5000.0 / quality
        else:
            scale = 200.0 - 2.0 * quality
        return torch.clamp(torch.floor((table * scale + 50) / 100), min=1)

    def forward(self, img: torch.Tensor, quality: int) -> torch.Tensor:
        """
        Apply JPEG compression simulation.
        img: [B, 3, H, W] in [0, 1]
        quality: JPEG quality factor (1-100, higher = less compression)
        Returns: [B, 3, H, W] in [0, 1]
        """
        B, C, H, W = img.shape

        # pad to multiple of 8
        ph = (8 - H % 8) % 8
        pw = (8 - W % 8) % 8
        if ph > 0 or pw > 0:
            img = F.pad(img, (0, pw, 0, ph), mode="reflect")
        Hp, Wp = img.shape[2], img.shape[3]
        nH, nW = Hp // 8, Wp // 8

        # RGB → YCbCr (values in [0, 255] range for JPEG math)
        img_255 = img * 255.0
        y = 0.299 * img_255[:, 0] + 0.587 * img_255[:, 1] + 0.114 * img_255[:, 2]
        cb = -0.169 * img_255[:, 0] - 0.331 * img_255[:, 1] + 0.500 * img_255[:, 2] + 128
        cr = 0.500 * img_255[:, 0] - 0.419 * img_255[:, 1] - 0.081 * img_255[:, 2] + 128

        # scale quantization tables for this quality
        lq = self._scale_table(self.luma_table, quality)
        cq = self._scale_table(self.chroma_table, quality)
        tables = [lq, cq, cq]  # Y, Cb, Cr

        # DCT matrices for batch matmul
        D = self.dct_mat.unsqueeze(0)    # [1, 8, 8]
        DT = self.idct_mat.unsqueeze(0)  # [1, 8, 8]

        result_channels = []
        for c, ch_data in enumerate([y, cb, cr]):
            # reshape to 8×8 blocks: [B, Hp, Wp] → [B*nH*nW, 8, 8]
            blocks = ch_data.reshape(B, nH, 8, nW, 8)
            blocks = blocks.permute(0, 1, 3, 2, 4).reshape(-1, 8, 8)

            # level shift
            blocks = blocks - 128.0

            # forward DCT: D @ blocks @ D^T
            blocks = D @ blocks @ DT

            # quantize + dequantize
            qt = tables[c].unsqueeze(0)  # [1, 8, 8]
            blocks = torch.round(blocks / qt) * qt

            # inverse DCT: D^T @ blocks @ D
            blocks = DT @ blocks @ D

            # level shift back
            blocks = blocks + 128.0

            # reshape back: [B*nH*nW, 8, 8] → [B, Hp, Wp]
            ch_data = blocks.reshape(B, nH, nW, 8, 8)
            ch_data = ch_data.permute(0, 1, 3, 2, 4).reshape(B, Hp, Wp)
            result_channels.append(ch_data)

        # YCbCr → RGB
        y_ch, cb_ch, cr_ch = result_channels
        cb_ch = cb_ch - 128
        cr_ch = cr_ch - 128
        r = y_ch + 1.402 * cr_ch
        g = y_ch - 0.344 * cb_ch - 0.714 * cr_ch
        b = y_ch + 1.772 * cb_ch
        rgb = torch.stack([r, g, b], dim=1) / 255.0

        return rgb[:, :, :H, :W].clamp(0, 1)


# ─────────────────────────────────────────
#  Sinc Filter
# ─────────────────────────────────────────

def apply_sinc_filter(img: torch.Tensor, cutoff: float, kernel_size: int = 21):
    """Apply sinc low-pass filter via convolution.
    cutoff: normalized cutoff frequency in (0, 1].
    """
    half = kernel_size // 2
    x = torch.arange(-half, half + 1, device=img.device, dtype=torch.float32)
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    r = torch.sqrt(xx ** 2 + yy ** 2).clamp(min=1e-8)

    # sinc kernel with Hamming window
    kernel = torch.sin(cutoff * math.pi * r) / (math.pi * r)
    kernel[half, half] = cutoff  # limit at r=0

    # Hamming window for smooth rolloff
    window = 0.54 - 0.46 * torch.cos(
        2 * math.pi * torch.arange(kernel_size, device=img.device) / (kernel_size - 1)
    )
    window_2d = window.unsqueeze(1) * window.unsqueeze(0)
    kernel = kernel * window_2d
    kernel = kernel / kernel.sum()

    # apply as depthwise conv
    C = img.shape[1]
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(C, -1, -1, -1)
    return F.conv2d(F.pad(img, [half] * 4, mode="reflect"), kernel, groups=C)


# ─────────────────────────────────────────
#  Real-ESRGAN Second-Order Degradation
# ─────────────────────────────────────────

class RealESRGANDegradation(nn.Module):
    """
    Second-order degradation pipeline from Real-ESRGAN (Wang et al., 2021).

    Applies two cascaded degradation rounds to HR images to produce
    realistic LR images with diverse artifacts:

        Round 1: blur → resize → noise → JPEG
        Round 2: blur → resize (to target LR) → noise → JPEG/sinc

    All operations run on GPU. Sampling is randomized per call.
    """

    def __init__(self, scale: int = 4):
        super().__init__()
        self.scale = scale
        self.jpeg = DiffJPEG()

        # ── first degradation parameters ──
        self.blur_kernels_1 = [7, 9, 11, 13, 15, 17, 19, 21]
        self.blur_sigma_1 = (0.2, 3.0)
        self.resize_range_1 = (0.15, 1.5)
        self.noise_sigma_1 = (0, 20)       # /255 scale
        self.jpeg_quality_1 = (30, 95)

        # ── second degradation parameters (lighter) ──
        self.blur_kernels_2 = [7, 9, 11, 13, 15, 17, 19, 21]
        self.blur_sigma_2 = (0.2, 1.5)
        self.noise_sigma_2 = (0, 15)
        self.jpeg_quality_2 = (30, 95)
        self.sinc_prob = 0.5              # prob of sinc vs JPEG in round 2

    @torch.no_grad()
    def forward(self, hr: torch.Tensor) -> torch.Tensor:
        """
        Generate degraded LR from HR batch.
        hr:  [B, 3, H, W] on GPU, values in [0, 1]
        Returns: [B, 3, H//scale, W//scale] on GPU, values in [0, 1]
        """
        B, C, H, W = hr.shape
        target_h, target_w = H // self.scale, W // self.scale

        out = hr

        # ── first degradation ──
        out = random_blur(out, self.blur_kernels_1, self.blur_sigma_1)
        out = random_resize(out, self.resize_range_1)
        out = add_gaussian_noise(out, self.noise_sigma_1)
        out = out.clamp(0, 1)
        quality_1 = random.randint(*self.jpeg_quality_1)
        out = self.jpeg(out, quality_1)

        # ── second degradation ──
        out = random_blur(out, self.blur_kernels_2, self.blur_sigma_2)
        # final resize to target LR resolution
        out = F.interpolate(
            out,
            size=(target_h, target_w),
            mode=random.choice(["bilinear", "bicubic", "area"]),
            antialias=True,
        )
        out = add_gaussian_noise(out, self.noise_sigma_2)
        out = out.clamp(0, 1)

        # JPEG or sinc filter
        if random.random() < self.sinc_prob:
            cutoff = random.uniform(0.3, 1.0)
            out = apply_sinc_filter(out, cutoff)
        else:
            quality_2 = random.randint(*self.jpeg_quality_2)
            out = self.jpeg(out, quality_2)

        return out.clamp(0, 1)
