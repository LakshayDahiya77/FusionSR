"""
FusionSR-v4 datasets.

Datasets:
    PairedSRDataset  — pre-computed LR+HR pairs (DIV2K, Flickr2K)
    HROnlyDataset    — HR-only (LSDIR); LR generated on GPU in trainer
    BenchmarkDataset — Set5/Set14 evaluation
"""

import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from PIL import Image


# ─────────────────────────────────────────
#  GPU Augmentation (geometric only)
# ─────────────────────────────────────────

def gpu_augment(lr: torch.Tensor, hr: torch.Tensor):
    """
    Random flip and rotation on GPU tensors.
    lr: [B, C, H, W]
    hr: [B, C, H*scale, W*scale]
    """
    if random.random() > 0.5:
        lr = torch.flip(lr, dims=[-1])
        hr = torch.flip(hr, dims=[-1])

    if random.random() > 0.5:
        lr = torch.flip(lr, dims=[-2])
        hr = torch.flip(hr, dims=[-2])

    k = random.randint(0, 3)
    if k > 0:
        lr = torch.rot90(lr, k, dims=[-2, -1])
        hr = torch.rot90(hr, k, dims=[-2, -1])

    return lr.contiguous(), hr.contiguous()


# ─────────────────────────────────────────
#  Paired SR Dataset (DIV2K / Flickr2K)
# ─────────────────────────────────────────

class PairedSRDataset(Dataset):
    """
    Pre-loads all LR images into RAM for speed. HR loaded lazily.
    Works for DIV2K and Flickr2K — expects matching filenames.
    HR: 0001.png  LR: 0001x4.png
    """

    def __init__(self, hr_dir: str, lr_dir: str, patch_lr: int = 128):
        super().__init__()
        self.patch_lr = patch_lr

        hr_files = sorted(Path(hr_dir).glob("*.png"))
        assert len(hr_files) > 0, f"No PNG files in {hr_dir}"

        print(f"pre-loading {len(hr_files)} LR images...", end=" ", flush=True)
        self.lr_images = []
        self.hr_paths = []

        for hr_path in hr_files:
            lr_path = Path(lr_dir) / f"{hr_path.stem}x4.png"
            if not lr_path.exists():
                # fallback: same filename (some datasets don't add x4 suffix)
                lr_path = Path(lr_dir) / hr_path.name
            lr = np.array(Image.open(lr_path).convert("RGB"), dtype=np.uint8)
            self.lr_images.append(lr)
            self.hr_paths.append(hr_path)

        print("done.")

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_np = self.lr_images[idx]
        hr_np = np.array(Image.open(self.hr_paths[idx]).convert("RGB"), dtype=np.uint8)
        lr_np, hr_np = self._random_crop(lr_np, hr_np)

        lr = torch.from_numpy(lr_np.copy()).permute(2, 0, 1).float().div_(255.0)
        hr = torch.from_numpy(hr_np.copy()).permute(2, 0, 1).float().div_(255.0)
        return lr, hr

    def _random_crop(self, lr: np.ndarray, hr: np.ndarray):
        """Random crop on uint8 numpy arrays [H, W, 3]."""
        h, w = lr.shape[:2]
        p = self.patch_lr

        if h < p or w < p:
            lr = np.pad(lr, ((0, max(0, p - h)), (0, max(0, p - w)), (0, 0)), mode='reflect')
            hr = np.pad(hr, ((0, max(0, (p - h) * 4)), (0, max(0, (p - w) * 4)), (0, 0)), mode='reflect')
            h, w = lr.shape[:2]

        x = random.randint(0, w - p)
        y = random.randint(0, h - p)

        lr = lr[y:y + p, x:x + p]
        hr = hr[y * 4:y * 4 + p * 4, x * 4:x * 4 + p * 4]
        return lr, hr


# ─────────────────────────────────────────
#  HR-Only Dataset (LSDIR, or any HR folder)
# ─────────────────────────────────────────

class HROnlyDataset(Dataset):
    """
    For datasets with only HR images (e.g., LSDIR).
    LR generated on-the-fly via bicubic downscale.
    Scans recursively for jpg/png/jpeg files.
    """

    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    def __init__(self, hr_dirs: list, patch_lr: int = 128, scale: int = 4):
        super().__init__()
        self.patch_lr = patch_lr
        self.patch_hr = patch_lr * scale
        self.scale = scale

        self.hr_files = []
        for d in hr_dirs:
            files = sorted(
                p for p in Path(d).rglob("*")
                if p.suffix.lower() in self.IMG_EXTENSIONS
            )
            self.hr_files.extend(files)

        assert len(self.hr_files) > 0, f"No images found in {hr_dirs}"
        print(f"HROnlyDataset: {len(self.hr_files)} images indexed")

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_np = np.array(
            Image.open(self.hr_files[idx]).convert("RGB"), dtype=np.uint8
        )

        # random crop at HR resolution
        hr_np = self._random_crop_hr(hr_np)

        hr = torch.from_numpy(hr_np.copy()).permute(2, 0, 1).float().div_(255.0)
        return hr

    def _random_crop_hr(self, hr: np.ndarray) -> np.ndarray:
        """Random crop at HR resolution [H, W, 3]."""
        h, w = hr.shape[:2]
        p = self.patch_hr

        if h < p or w < p:
            hr = np.pad(
                hr,
                ((0, max(0, p - h)), (0, max(0, p - w)), (0, 0)),
                mode='reflect',
            )
            h, w = hr.shape[:2]

        x = random.randint(0, w - p)
        y = random.randint(0, h - p)
        return hr[y:y + p, x:x + p]


# ─────────────────────────────────────────
#  Benchmark Dataset (Set5, Set14)
# ─────────────────────────────────────────

class BenchmarkDataset(Dataset):
    """
    Standard SR benchmark datasets (Set5, Set14).
    HR: GTmod12 folder, LR: LRbicx4 folder.
    Returns (lr, hr, filename) for per-image logging.
    """

    def __init__(self, hr_dir: str, lr_dir: str):
        super().__init__()
        self.hr_files = sorted(Path(hr_dir).glob("*.png"))
        self.lr_dir = Path(lr_dir)
        assert len(self.hr_files) > 0, f"No PNG files in {hr_dir}"

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        lr_path = self.lr_dir / hr_path.name

        hr = np.array(Image.open(hr_path).convert("RGB"), dtype=np.uint8)
        lr = np.array(Image.open(lr_path).convert("RGB"), dtype=np.uint8)

        hr = torch.from_numpy(hr).permute(2, 0, 1).float() / 255.0
        lr = torch.from_numpy(lr).permute(2, 0, 1).float() / 255.0

        return lr, hr, hr_path.name


# ─────────────────────────────────────────
#  Dataloader Factories
# ─────────────────────────────────────────

def make_train_dl(
    hr_dirs: list,
    lr_dirs: list = None,
    patch_lr: int = 128,
    batch_size: int = 16,
    num_workers: int = 4,
    scale: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """Create training dataloader.

    If lr_dirs provided: load pre-computed LR-HR pairs (DIV2K/Flickr2K).
    If lr_dirs is None or empty: HR-only mode, generate LR on-the-fly (LSDIR).
    """
    if lr_dirs:
        # paired LR-HR datasets
        datasets = []
        for hr_dir, lr_dir in zip(hr_dirs, lr_dirs):
            datasets.append(PairedSRDataset(hr_dir, lr_dir, patch_lr=patch_lr))

        if len(datasets) == 1:
            ds = datasets[0]
        else:
            ds = ConcatDataset(datasets)
            print(f"combined dataset: {len(ds)} images "
                  f"({' + '.join(str(len(d)) for d in datasets)})")
    else:
        # HR-only — LR generated on-the-fly
        ds = HROnlyDataset(hr_dirs, patch_lr=patch_lr, scale=scale)

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def make_benchmark_dl(hr_dir: str, lr_dir: str) -> DataLoader:
    """Set5 / Set14 benchmark dataloader."""
    ds = BenchmarkDataset(hr_dir, lr_dir)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)


def generate_lr_on_gpu(hr: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """
    Generate LR from HR via bicubic downscaling on GPU.
    hr: [B, 3, H, W] on CUDA
    returns: [B, 3, H//scale, W//scale] on CUDA
    """
    return F.interpolate(
        hr,
        scale_factor=1.0 / scale,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(0, 1)


def generate_lr_on_gpu(hr: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """
    Generate LR from HR via bicubic downscaling on GPU.
    hr: [B, 3, H, W] on CUDA
    returns: [B, 3, H//scale, W//scale] on CUDA
    """
    return F.interpolate(
        hr,
        scale_factor=1.0 / scale,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(0, 1)
