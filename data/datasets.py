import os
import shutil
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image


# ─────────────────────────────────────────
#  RAM Disk Setup
# ─────────────────────────────────────────
def setup_ramdisk(src_dirs: dict, ramdisk: str = "/dev/shm/div2k") -> dict:
    os.makedirs(ramdisk, exist_ok=True)
    dst_dirs = {}
    for name, src in src_dirs.items():
        dst = os.path.join(ramdisk, name)
        if os.path.exists(dst):
            print(f"{name}: already in RAM disk")
            dst_dirs[name] = dst
            continue
        print(f"copying {name}...", end=" ")
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
    lr: [B, 3, H, W]
    hr: [B, 3, H*4, W*4]
    """
    # random horizontal flip
    if random.random() > 0.5:
        lr = torch.flip(lr, dims=[-1])
        hr = torch.flip(hr, dims=[-1])

    # random vertical flip
    if random.random() > 0.5:
        lr = torch.flip(lr, dims=[-2])
        hr = torch.flip(hr, dims=[-2])

    # random 90-degree rotation (0, 90, 180, 270)
    k = random.randint(0, 3)
    if k > 0:
        lr = torch.rot90(lr, k, dims=[-2, -1])
        hr = torch.rot90(hr, k, dims=[-2, -1])

    return lr, hr


# ─────────────────────────────────────────
#  Fast Dataset — pre-loads all images into RAM
# ─────────────────────────────────────────
class DIV2KDatasetFast(Dataset):
    """
    Pre-loads all LR images into RAM (small, safe).
    HR images loaded lazily from RAM disk (fast enough — already in /dev/shm).

    """

    def __init__(self, hr_dir, lr_dir, patch_lr=64, training=True):
        super().__init__()
        self.patch_lr = patch_lr
        self.training = training
        self.hr_dir = Path(hr_dir)

        hr_files = sorted(Path(hr_dir).glob("*.png"))
        assert len(hr_files) > 0

        print(f"pre-loading {len(hr_files)} LR images...", end=" ", flush=True)
        self.lr_images = []
        self.hr_paths = []

        for hr_path in hr_files:
            lr_path = Path(lr_dir) / f"{hr_path.stem}x4.png"
            # pre-load LR only — small images, safe
            lr = np.array(Image.open(lr_path).convert("RGB"), dtype=np.float32) / 255.0
            self.lr_images.append(lr)
            self.hr_paths.append(hr_path)  # HR loaded lazily

        print(f"done.")

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr = torch.from_numpy(self.lr_images[idx]).permute(2, 0, 1)

        # load HR from RAM disk lazily — fast since /dev/shm is in memory
        hr = (
            np.array(Image.open(self.hr_paths[idx]).convert("RGB"), dtype=np.float32)
            / 255.0
        )
        hr = torch.from_numpy(hr).permute(2, 0, 1)

        if self.training:
            lr, hr = self._random_crop(lr, hr)

        return lr, hr

    def _random_crop(self, lr, hr):
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
#  Original Dataset (PIL-based, kept for reference)
# ─────────────────────────────────────────
class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_lr=64, training=True):
        super().__init__()
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.patch_lr = patch_lr
        self.training = training
        self.hr_files = sorted(self.hr_dir.glob("*.png"))
        assert len(self.hr_files) > 0

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        lr_path = self.lr_dir / f"{hr_path.stem}x4.png"
        hr = np.array(Image.open(hr_path).convert("RGB"), dtype=np.float32) / 255.0
        lr = np.array(Image.open(lr_path).convert("RGB"), dtype=np.float32) / 255.0
        hr = torch.from_numpy(hr).permute(2, 0, 1)
        lr = torch.from_numpy(lr).permute(2, 0, 1)
        if self.training:
            lr, hr = self._random_crop(lr, hr)
        return lr, hr

    def _random_crop(self, lr, hr):
        _, h, w = lr.shape
        p = self.patch_lr
        if h < p or w < p:
            lr = F.pad(lr, (0, max(0, p - w), 0, max(0, p - h)))
            hr = F.pad(hr, (0, max(0, p - w) * 4, 0, max(0, p - h) * 4))
            _, h, w = lr.shape
        x = random.randint(0, w - p)
        y = random.randint(0, h - p)
        lr = lr[:, y : y + p, x : x + p]
        hr = hr[:, y * 4 : y * 4 + p * 4, x * 4 : x * 4 + p * 4]
        return lr, hr


# ─────────────────────────────────────────
#  Dataloaders
# ─────────────────────────────────────────


def make_train_dataloader(
    train_hr: str,
    train_lr: str,
    patch_lr: int = 64,
    batch_size: int = 32,
    num_workers: int = 4,
):
    train_ds = DIV2KDatasetFast(train_hr, train_lr, patch_lr=patch_lr, training=True)
    return DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )


class BenchmarkDataset(Dataset):
    """
    Dataset for standard SR benchmarks (Set5, Set14).
    Uses pre-paired LR/HR images from the benchmark dataset.
    HR: GTmod12 folder
    LR: LRbicx4 folder
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
        lr_path = self.lr_dir / hr_path.name  # same filename in LR folder

        hr = np.array(Image.open(hr_path).convert("RGB"), dtype=np.float32) / 255.0
        lr = np.array(Image.open(lr_path).convert("RGB"), dtype=np.float32) / 255.0

        hr = torch.from_numpy(hr).permute(2, 0, 1)
        lr = torch.from_numpy(lr).permute(2, 0, 1)

        return lr, hr, hr_path.name  # return filename for logging


def make_benchmark_loader(hr_dir: str, lr_dir: str):
    ds = BenchmarkDataset(hr_dir, lr_dir)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
