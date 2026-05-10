import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(max_val**2 / mse).item()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    max_val: float = 1.0,
) -> float:
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    kernel = _gaussian_kernel(window_size, sigma=1.5).to(pred.device)
    kernel = kernel.expand(pred.shape[1], 1, window_size, window_size)
    pad = window_size // 2
    mu1 = F.conv2d(pred, kernel, padding=pad, groups=pred.shape[1])
    mu2 = F.conv2d(target, kernel, padding=pad, groups=pred.shape[1])
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = (
        F.conv2d(pred * pred, kernel, padding=pad, groups=pred.shape[1]) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(target * target, kernel, padding=pad, groups=pred.shape[1]) - mu2_sq
    )
    sigma12 = (
        F.conv2d(pred * target, kernel, padding=pad, groups=pred.shape[1]) - mu1_mu2
    )
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()


def rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB [B, 3, H, W] to Y channel [B, 1, H, W].
    Standard coefficients used in SR papers (ITU-R BT.601).
    Input values in [0, 1].
    """
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return (
        16.0 / 255.0
        + (65.481 / 255.0) * r
        + (128.553 / 255.0) * g
        + (24.966 / 255.0) * b
    )


def psnr_y(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR on Y channel — matches published SR paper methodology."""
    return psnr(rgb_to_y(pred), rgb_to_y(target))


def ssim_y(pred: torch.Tensor, target: torch.Tensor) -> float:
    """SSIM on Y channel — matches published SR paper methodology."""
    return ssim(rgb_to_y(pred), rgb_to_y(target))


def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = g.outer(g)
    return kernel.unsqueeze(0).unsqueeze(0)
