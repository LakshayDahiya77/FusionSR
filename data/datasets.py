import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image


class DIV2KDataset(Dataset):
    """
    DIV2K dataset for 4x SR training.

    HR images live in hr_dir as 0001.png ... 0800.png
    LR images live in lr_dir as 0001x4.png ... 0800x4.png

    During training:
        - randomly crops a patch of size (patch_lr x patch_lr) from LR
        - crops the corresponding (patch_lr*4 x patch_lr*4) region from HR
        - applies random horizontal flip and random 90-degree rotation (augmentation)

    During validation:
        - returns full images, no cropping or augmentation
        - used to compute PSNR / SSIM on complete images

    """

    def __init__(
        self,
        hr_dir: str,
        lr_dir: str,
        patch_lr: int = 64,  # LR patch size, HR patch = patch_lr * 4
        training: bool = True,
    ):
        super().__init__()
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.patch_lr = patch_lr
        self.training = training

        # collect all HR filenames, derive LR names from them
        self.hr_files = sorted(self.hr_dir.glob("*.png"))
        assert len(self.hr_files) > 0, f"No PNG files found in {hr_dir}"

    def __len__(self) -> int:
        return len(self.hr_files)

    def __getitem__(self, idx: int):
        hr_path = self.hr_files[idx]

        # derive LR filename: 0001.png → 0001x4.png
        stem = hr_path.stem  # "0001"
        lr_path = self.lr_dir / f"{stem}x4.png"

        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")

        if self.training:
            lr, hr = self._random_crop(lr, hr)
            lr, hr = self._augment(lr, hr)

        # PIL → tensor, values in [0, 1]
        lr = TF.to_tensor(lr)  # [3, H, W]
        hr = TF.to_tensor(hr)  # [3, H*4, W*4]

        return lr, hr

    def _random_crop(self, lr: Image.Image, hr: Image.Image):
        lr_w, lr_h = lr.size
        p = self.patch_lr

        # make sure image is large enough
        if lr_w < p or lr_h < p:
            lr = TF.resize(
                lr,
                (max(lr_h, p), max(lr_w, p)),
                interpolation=TF.InterpolationMode.BICUBIC,
            )
            hr = TF.resize(
                hr,
                (max(lr_h, p) * 4, max(lr_w, p) * 4),
                interpolation=TF.InterpolationMode.BICUBIC,
            )
            lr_w, lr_h = lr.size

        # random top-left corner in LR space
        x = random.randint(0, lr_w - p)
        y = random.randint(0, lr_h - p)

        lr_crop = TF.crop(lr, y, x, p, p)
        hr_crop = TF.crop(hr, y * 4, x * 4, p * 4, p * 4)

        return lr_crop, hr_crop

    def _augment(self, lr: Image.Image, hr: Image.Image):
        # random horizontal flip
        if random.random() > 0.5:
            lr = TF.hflip(lr)
            hr = TF.hflip(hr)

        # random 90-degree rotation (0, 90, 180, 270)
        k = random.randint(0, 3)
        if k > 0:
            lr = TF.rotate(lr, 90 * k)
            hr = TF.rotate(hr, 90 * k)

        return lr, hr


def make_dataloaders(
    train_hr: str,
    train_lr: str,
    valid_hr: str,
    valid_lr: str,
    patch_lr: int = 64,
    batch_size: int = 16,
    num_workers: int = 4,
):
    train_ds = DIV2KDataset(train_hr, train_lr, patch_lr=patch_lr, training=True)
    valid_ds = DIV2KDataset(valid_hr, valid_lr, patch_lr=patch_lr, training=False)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=1,  # full images, one at a time
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_dl, valid_dl
