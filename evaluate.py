"""
evaluate.py — Benchmark FusionSR models on standard SR datasets.

Usage:
    python evaluate.py --model classical --checkpoint /path/to/checkpoint.pt
    python evaluate.py --model satellite --checkpoint /path/to/checkpoint.pt
    python evaluate.py --model both

Outputs:
    results/benchmark_results.csv
    results/benchmark_results.json
    results/sample_images/  (side-by-side comparisons)
"""

import os
import json
import argparse
import torch
import wandb
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from kaggle_secrets import UserSecretsClient
from huggingface_hub import hf_hub_download
from models.fusionsr import FusionSR
from utils.metrics import psnr, ssim, psnr_y, ssim_y, rgb_to_y

# ─────────────────────────────────────────
#  Config
# ─────────────────────────────────────────
BENCH_BASE = "/kaggle/input/datasets/jesucristo/super-resolution-benchmarks"
DIOR_BASE = (
    "/kaggle/input/datasets/redzapdos123/dior-r-dataset-yolov11-obb-format/YOLODIOR-R"
)

MODEL_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "channels": 96,
    "num_groups": 6,
    "num_rcab": 6,
    "window_size": 8,
    "num_heads": 4,
    "scale": 4,
}

CLASSICAL_BENCHMARKS = {
    "Set5": (
        f"{BENCH_BASE}/Set5/Set5/GTmod12",
        f"{BENCH_BASE}/Set5/Set5/LRbicx4",
        "paired",
    ),
    "Set14": (
        f"{BENCH_BASE}/Set14/Set14/GTmod12",
        f"{BENCH_BASE}/Set14/Set14/LRbicx4",
        "paired",
    ),
    "BSD100": (f"{BENCH_BASE}/BSD68/BSD68", None, "hr_only"),
    "Urban100": (f"{BENCH_BASE}/urban100/urban100", None, "hr_only"),
}

SATELLITE_BENCHMARKS = {
    "DIOR-test": (f"{DIOR_BASE}/test/images", None, "hr_only"),
    "Urban100": (f"{BENCH_BASE}/urban100/urban100", None, "hr_only"),
}


# ─────────────────────────────────────────
#  Datasets
# ─────────────────────────────────────────
class PairedDataset(Dataset):
    """Pre-paired LR/HR dataset (Set5, Set14)."""

    def __init__(self, hr_dir, lr_dir):
        self.hr_files = sorted(Path(hr_dir).glob("*.png"))
        self.lr_dir = Path(lr_dir)
        assert len(self.hr_files) > 0

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        lr_path = self.lr_dir / hr_path.name
        hr = torch.from_numpy(
            np.array(Image.open(hr_path).convert("RGB"), dtype=np.float32) / 255.0
        ).permute(2, 0, 1)
        lr = torch.from_numpy(
            np.array(Image.open(lr_path).convert("RGB"), dtype=np.float32) / 255.0
        ).permute(2, 0, 1)
        return lr, hr, hr_path.name


class HROnlyDataset(Dataset):
    """HR-only dataset — generates LR via bicubic (BSD100, Urban100, DIOR)."""

    def __init__(self, hr_dir, scale=4, extensions=(".png", ".jpg", ".jpeg")):
        self.hr_files = sorted(
            [p for p in Path(hr_dir).rglob("*") if p.suffix.lower() in extensions]
        )
        self.scale = scale
        assert len(self.hr_files) > 0

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        hr = torch.from_numpy(
            np.array(Image.open(hr_path).convert("RGB"), dtype=np.float32) / 255.0
        ).permute(2, 0, 1)
        # generate LR via bicubic
        lr = (
            F.interpolate(
                hr.unsqueeze(0),
                scale_factor=1.0 / self.scale,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            .squeeze(0)
            .clamp(0, 1)
        )
        return lr, hr, hr_path.name


# ─────────────────────────────────────────
#  Model loader
# ─────────────────────────────────────────
def load_model(model_identifier: str, device: torch.device) -> torch.nn.Module:
    """Loads from HF Hub if given 'classical'/'satellite', else treats as local path."""
    if model_identifier in ["classical", "satellite"]:
        filename = f"fusionsr-v2-{model_identifier}.pt"
        ckpt_path = hf_hub_download(
            repo_id="lakshaydahiya/FusionSR-v2",
            filename=filename,
        )
        print(f"loaded {filename} from HuggingFace Hub")
    else:
        ckpt_path = model_identifier
        print(f"loaded local checkpoint from {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = FusionSR(**MODEL_CONFIG).to(device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()
    return model


# ─────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────
@torch.no_grad()
def run_model(model, lr_imgs, device, window_size=8):
    """Run model on LR batch, pad to window size, crop output."""
    lr_imgs = lr_imgs.to(device)
    _, _, h, w = lr_imgs.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        lr_imgs = F.pad(lr_imgs, (0, pad_w, 0, pad_h), mode="reflect")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(lr_imgs).float().clamp(0, 1)
    return pred[:, :, : h * 4, : w * 4]


# ─────────────────────────────────────────
#  Bicubic baseline
# ─────────────────────────────────────────
def bicubic_upsample(lr_imgs, scale=4):
    _, _, h, w = lr_imgs.shape
    return F.interpolate(
        lr_imgs,
        size=(h * scale, w * scale),
        mode="bicubic",
        align_corners=False,
    ).clamp(0, 1)


# ─────────────────────────────────────────
#  Evaluate one dataset
# ─────────────────────────────────────────
def evaluate_dataset(
    model,
    dataset_name: str,
    hr_dir: str,
    lr_dir: str,
    mode: str,  # "paired" | "hr_only"
    device: torch.device,
    scale: int = 4,
    save_samples: bool = True,
    sample_dir: str = "results/samples",
    n_samples: int = 3,
) -> dict:

    if mode == "paired":
        ds = PairedDataset(hr_dir, lr_dir)
    else:
        ds = HROnlyDataset(hr_dir, scale=scale)

    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    model_psnr_rgb = model_ssim_rgb = 0.0
    model_psnr_y = model_ssim_y = 0.0
    bic_psnr_rgb = bic_ssim_rgb = 0.0
    bic_psnr_y = bic_ssim_y = 0.0

    os.makedirs(sample_dir, exist_ok=True)
    saved = 0

    for i, (lr_imgs, hr_imgs, fname) in enumerate(dl):
        hr_imgs = hr_imgs.to(device).float()
        lr_imgs = lr_imgs.to(device).float()

        # model prediction
        pred = run_model(model, lr_imgs, device)

        # bicubic baseline
        bic = bicubic_upsample(lr_imgs, scale=scale)

        # boundary crop — standard SR evaluation
        b = scale
        pred = pred[:, :, b:-b, b:-b]
        bic = bic[:, :, b:-b, b:-b]
        hr_crop = hr_imgs[:, :, b:-b, b:-b]

        # metrics
        model_psnr_rgb += psnr(pred, hr_crop)
        model_ssim_rgb += ssim(pred, hr_crop)
        model_psnr_y += psnr_y(pred, hr_crop)
        model_ssim_y += ssim_y(pred, hr_crop)
        bic_psnr_rgb += psnr(bic, hr_crop)
        bic_ssim_rgb += ssim(bic, hr_crop)
        bic_psnr_y += psnr_y(bic, hr_crop)
        bic_ssim_y += ssim_y(bic, hr_crop)

        # save sample comparisons
        if save_samples and saved < n_samples:
            save_comparison(
                lr=lr_imgs[0].cpu(),
                sr=pred[0].cpu(),
                hr=hr_crop[0].cpu(),
                bic=bic[0].cpu(),
                filename=f"{sample_dir}/{dataset_name}_{fname[0]}",
                scale=scale,
            )
            saved += 1

    n = len(dl)
    return {
        "dataset": dataset_name,
        "n_images": n,
        "model_psnr_rgb": round(model_psnr_rgb / n, 4),
        "model_ssim_rgb": round(model_ssim_rgb / n, 4),
        "model_psnr_y": round(model_psnr_y / n, 4),
        "model_ssim_y": round(model_ssim_y / n, 4),
        "bic_psnr_rgb": round(bic_psnr_rgb / n, 4),
        "bic_ssim_rgb": round(bic_ssim_rgb / n, 4),
        "bic_psnr_y": round(bic_psnr_y / n, 4),
        "bic_ssim_y": round(bic_ssim_y / n, 4),
    }


# ─────────────────────────────────────────
#  Save comparison image
# ─────────────────────────────────────────
def save_comparison(lr, sr, hr, bic, filename, scale=4):
    """Save side-by-side: bicubic | FusionSR | HR"""
    h, w = hr.shape[-2], hr.shape[-1]

    lr_up = (
        F.interpolate(lr.unsqueeze(0), size=(h, w), mode="bicubic", align_corners=False)
        .squeeze(0)
        .clamp(0, 1)
    )

    # match sizes
    sr = sr[:, :h, :w]
    bic = bic[:, :h, :w]

    # concat horizontally: bicubic | FusionSR | HR
    comparison = torch.cat([bic, sr, hr], dim=2)
    img = (comparison.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(filename)


# ─────────────────────────────────────────
#  Export results
# ─────────────────────────────────────────
def export_results(results: list, model_name: str, out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)

    # JSON
    json_path = f"{out_dir}/{model_name}_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # CSV
    import csv

    csv_path = f"{out_dir}/{model_name}_results.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # Pretty print table
    print(f"\n{'─'*90}")
    print(
        f"{'Dataset':<12} {'N':>5} {'Bic PSNR-Y':>11} {'Bic SSIM-Y':>11} {'Model PSNR-Y':>13} {'Model SSIM-Y':>13}"
    )
    print(f"{'─'*90}")
    for r in results:
        print(
            f"{r['dataset']:<12} {r['n_images']:>5} "
            f"{r['bic_psnr_y']:>11.2f} {r['bic_ssim_y']:>11.4f} "
            f"{r['model_psnr_y']:>13.2f} {r['model_ssim_y']:>13.4f}"
        )
    print(f"{'─'*90}")
    print(f"\nsaved: {json_path}")
    print(f"saved: {csv_path}")

    return json_path, csv_path


# ─────────────────────────────────────────
#  Main
# ─────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # W&B login
    try:
        from kaggle_secrets import UserSecretsClient

        secrets = UserSecretsClient()
        wandb.login(key=secrets.get_secret("WANDB_API_KEY"))
    except Exception:
        pass  # running locally without Kaggle secrets

    models_to_eval = []

    if args.model in ("classical", "both"):
        # If no local checkpoint provided, pass "classical" to trigger HF download
        ckpt = args.classical_checkpoint or "classical"
        print(f"\nloading FusionSR-v2-Classical...")
        m = load_model(ckpt, device)
        models_to_eval.append(("FusionSR-v2-Classical", m, CLASSICAL_BENCHMARKS))

    if args.model in ("satellite", "both"):
        ckpt = args.satellite_checkpoint or "satellite"
        print(f"\nloading FusionSR-v2-Satellite...")
        m = load_model(ckpt, device)
        models_to_eval.append(("FusionSR-v2-Satellite", m, SATELLITE_BENCHMARKS))

    for model_name, model, benchmarks in models_to_eval:
        print(f"\n{'═'*50}")
        print(f"evaluating {model_name}")
        print(f"{'═'*50}")

        results = []
        for dataset_name, (hr_dir, lr_dir, mode) in benchmarks.items():
            print(f"\n→ {dataset_name} ({mode})")
            r = evaluate_dataset(
                model=model,
                dataset_name=dataset_name,
                hr_dir=hr_dir,
                lr_dir=lr_dir,
                mode=mode,
                device=device,
                save_samples=True,
                sample_dir=f"results/samples/{model_name}",
            )
            results.append(r)
            print(
                f"  PSNR-Y: {r['model_psnr_y']:.2f}dB | SSIM-Y: {r['model_ssim_y']:.4f} "
                f"(bicubic: {r['bic_psnr_y']:.2f}dB)"
            )

        export_results(results, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["classical", "satellite", "both"], default="both"
    )
    parser.add_argument("--classical_checkpoint", type=str, default=None)
    parser.add_argument("--satellite_checkpoint", type=str, default=None)
    args = parser.parse_args()
    main(args)
