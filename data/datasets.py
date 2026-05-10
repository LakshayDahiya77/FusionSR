import os
import shutil
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from PIL import Image


# ─────────────────────────────────────────
#  RAM Disk Setup
# ─────────────────────────────────────────
def setup_ramdisk(src_dirs: dict, ramdisk: str = "/dev/shm/fusionsr") -> dict:
    os.makedirs(ramdisk, exist_ok=True)
    dst_dirs = {}
    for name, src in src_dirs.items():
        dst = os.path.join(ramdisk, name)
        if os.path.exists(dst):
            print(f"{name}: already in RAM disk")
            dst_dirs[name] = dst
            continue
        print(f"copying {name}...", end=" ", flush=True)
        shutil.copytree(src, dst)
        print(f"done ({len(os.listdir(dst))} files)")
        dst_dirs[name] = dst
    stat = shutil.disk_usage("/dev/shm")
    print(f"RAM disk: {stat.used/1024**3:.2f}GB / {stat.total/1024**3:.2f}GB")
    return dst_dirs


# ─────────────────────────────────────────
#  GPU Augmentation
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

    return lr, hr


# ─────────────────────────────────────────
#  Fast Dataset — pre-loads LR into RAM, HR lazily
# ─────────────────────────────────────────
class DIV2KDatasetFast(Dataset):
    """
    Pre-loads all LR images into RAM.
    HR images loaded lazily from hr_dir (RAM disk or SSD).
    Works for both DIV2K and Flickr2K — same filename convention.
    HR: 000001.png  LR: 000001x4.png
    """

    def __init__(
        self, hr_dir: str, lr_dir: str, patch_lr: int = 64, training: bool = True
    ):
        super().__init__()
        self.patch_lr = patch_lr
        self.training = training

        hr_files = sorted(Path(hr_dir).glob("*.png"))
        assert len(hr_files) > 0, f"No PNG files in {hr_dir}"

        print(f"pre-loading {len(hr_files)} LR images...", end=" ", flush=True)
        self.lr_images = []
        self.hr_paths = []

        for hr_path in hr_files:
            lr_path = Path(lr_dir) / f"{hr_path.stem}x4.png"
            lr = np.array(Image.open(lr_path).convert("RGB"), dtype=np.float32) / 255.0
            self.lr_images.append(lr)
            self.hr_paths.append(hr_path)

        print("done.")

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr = torch.from_numpy(self.lr_images[idx]).permute(2, 0, 1)
        hr = (
            np.array(Image.open(self.hr_paths[idx]).convert("RGB"), dtype=np.float32)
            / 255.0
        )
        hr = torch.from_numpy(hr).permute(2, 0, 1)

        if self.training:
            lr, hr = self._random_crop(lr, hr)

        return lr, hr

    def _random_crop(self, lr: torch.Tensor, hr: torch.Tensor):
        _, h, w = lr.shape
        p = self.patch_lr

        if h < p or w < p:
            lr = F.pad(lr, (0, max(0, p - w), 0, max(0, p - h)))
            hr = F.pad(hr, (0, max(0, p - w) * 4, 0, max(0, p - h) * 4))
            _, h, w = lr.shape

        x = torch.randint(0, w - p + 1, (1,)).item()
        y = torch.randint(0, h - p + 1, (1,)).item()

        lr = lr[:, y : y + p, x : x + p]
        hr = hr[:, y * 4 : y * 4 + p * 4, x * 4 : x * 4 + p * 4]
        return lr, hr


# ─────────────────────────────────────────
#  Satellite Dataset — for PROBA-V / WorldStrat
# ─────────────────────────────────────────
class SatelliteDataset(Dataset):
    """
    Dataset for satellite image SR fine-tuning.
    Expects paired LR/HR satellite images.
    HR and LR in separate directories, matched by filename.

    PROBA-V structure:
        hr_dir/: 0001.png, 0002.png ...  (100m resolution)
        lr_dir/: 0001.png, 0002.png ...  (300m resolution, 3x downscaled)

    Set training=False for validation (returns full images).
    """

    def __init__(
        self,
        hr_dir: str,
        lr_dir: str,
        patch_lr: int = 64,
        training: bool = True,
    ):
        super().__init__()
        self.patch_lr = patch_lr
        self.training = training

        self.hr_files = sorted(Path(hr_dir).glob("*.png"))
        self.lr_dir = Path(lr_dir)
        assert len(self.hr_files) > 0, f"No PNG files in {hr_dir}"

        print(f"satellite dataset: {len(self.hr_files)} image pairs from {hr_dir}")

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        lr_path = self.lr_dir / hr_path.name  # same filename in LR dir

        hr = np.array(Image.open(hr_path).convert("RGB"), dtype=np.float32) / 255.0
        lr = np.array(Image.open(lr_path).convert("RGB"), dtype=np.float32) / 255.0

        hr = torch.from_numpy(hr).permute(2, 0, 1)
        lr = torch.from_numpy(lr).permute(2, 0, 1)

        if self.training:
            lr, hr = self._random_crop(lr, hr)

        return lr, hr

    def _random_crop(self, lr: torch.Tensor, hr: torch.Tensor):
        _, h, w = lr.shape
        p = self.patch_lr

        if h < p or w < p:
            lr = F.pad(lr, (0, max(0, p - w), 0, max(0, p - h)))
            hr = F.pad(hr, (0, max(0, p - w) * 4, 0, max(0, p - h) * 4))
            _, h, w = lr.shape

        x = torch.randint(0, w - p + 1, (1,)).item()
        y = torch.randint(0, h - p + 1, (1,)).item()

        lr = lr[:, y : y + p, x : x + p]
        hr = hr[:, y * 4 : y * 4 + p * 4, x * 4 : x * 4 + p * 4]
        return lr, hr


# ─────────────────────────────────────────
#  Benchmark Dataset — Set5, Set14
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

        hr = np.array(Image.open(hr_path).convert("RGB"), dtype=np.float32) / 255.0
        lr = np.array(Image.open(lr_path).convert("RGB"), dtype=np.float32) / 255.0

        hr = torch.from_numpy(hr).permute(2, 0, 1)
        lr = torch.from_numpy(lr).permute(2, 0, 1)

        return lr, hr, hr_path.name


# ─────────────────────────────────────────
#  Dataloader factories
# ─────────────────────────────────────────
def make_train_dataloader(
    train_hr: str,
    train_lr: str,
    patch_lr: int = 64,
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoader:
    """DIV2K only training dataloader."""
    ds = DIV2KDatasetFast(train_hr, train_lr, patch_lr=patch_lr, training=True)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )


def make_combined_dataloader(
    div2k_hr: str,
    div2k_lr: str,
    flickr_hr: str,
    flickr_lr: str,
    patch_lr: int = 64,
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoader:
    """
    Combined DIV2K + Flickr2K training dataloader.
    DIV2K: 800 images, Flickr2K: 2650 images → 3450 total.
    """
    div2k_ds = DIV2KDatasetFast(div2k_hr, div2k_lr, patch_lr=patch_lr, training=True)
    flickr_ds = DIV2KDatasetFast(flickr_hr, flickr_lr, patch_lr=patch_lr, training=True)
    combined = ConcatDataset([div2k_ds, flickr_ds])

    print(
        f"combined dataset: {len(combined)} images "
        f"({len(div2k_ds)} DIV2K + {len(flickr_ds)} Flickr2K)"
    )

    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )


def make_satellite_dataloader(
    train_hr: str,
    train_lr: str,
    patch_lr: int = 64,
    batch_size: int = 16,
    num_workers: int = 4,
) -> DataLoader:
    """Satellite image training dataloader (PROBA-V / WorldStrat)."""
    ds = SatelliteDataset(train_hr, train_lr, patch_lr=patch_lr, training=True)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )


def make_benchmark_loader(hr_dir: str, lr_dir: str) -> DataLoader:
    """Set5 / Set14 benchmark dataloader."""
    ds = BenchmarkDataset(hr_dir, lr_dir)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)


def make_satellite_val_loader(hr_dir: str, lr_dir: str) -> DataLoader:
    """Satellite validation dataloader — full images, no cropping."""
    ds = SatelliteDataset(hr_dir, lr_dir, training=False)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)


class SatelliteHRDataset(Dataset):
    """
    Dataset for satellite imagery SR fine-tuning.
    Loads HR images only — LR generated on GPU in training loop.
    Pre-loads all images into RAM for fast access.
    """

    def __init__(
        self,
        hr_dir: str,
        patch_hr: int = 256,
        training: bool = True,
        extensions: tuple = (".jpg", ".png", ".jpeg"),
    ):
        super().__init__()
        self.patch_hr = patch_hr
        self.training = training

        hr_files = sorted(
            [p for p in Path(hr_dir).rglob("*") if p.suffix.lower() in extensions]
        )
        assert len(hr_files) > 0, f"No images found in {hr_dir}"

        print(
            f"pre-loading {len(hr_files)} satellite HR images...", end=" ", flush=True
        )
        self.hr_images = []
        for p in hr_files:
            hr = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
            self.hr_images.append(hr)
        print("done.")

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr = torch.from_numpy(self.hr_images[idx]).permute(2, 0, 1)

        if self.training:
            hr = self._random_crop(hr)

        return hr  # LR generated on GPU in training loop

    def _random_crop(self, hr: torch.Tensor) -> torch.Tensor:
        _, h, w = hr.shape
        p = self.patch_hr

        if h < p or w < p:
            hr = F.pad(hr, (0, max(0, p - w), 0, max(0, p - h)))
            _, h, w = hr.shape

        x = torch.randint(0, w - p + 1, (1,)).item()
        y = torch.randint(0, h - p + 1, (1,)).item()
        return hr[:, y : y + p, x : x + p]


def make_satellite_hr_dataloader(
    hr_dir: str,
    patch_hr: int = 256,
    scale: int = 4,
    batch_size: int = 16,
    num_workers: int = 4,
    training: bool = True,
) -> DataLoader:
    ds = SatelliteHRDataset(
        hr_dir=hr_dir,
        patch_hr=patch_hr,
        training=training,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=training,
        persistent_workers=True if num_workers > 0 else False,
    )


def make_satellite_hr_dataloader(
    hr_dir: str,
    patch_hr: int = 256,
    scale: int = 4,
    batch_size: int = 16,
    num_workers: int = 4,
    training: bool = True,
) -> DataLoader:
    """
    Dataloader for satellite HR-only datasets with synthetic LR generation.
    Use for DIOR, EuroSAT, UC Merced etc.
    """
    ds = SatelliteHRDataset(
        hr_dir=hr_dir,
        patch_hr=patch_hr,
        scale=scale,
        training=training,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=training,
        persistent_workers=True if num_workers > 0 else False,
    )


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
