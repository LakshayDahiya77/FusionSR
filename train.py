"""
FusionSR-v3 training entry point.

Usage (Kaggle notebook Cell 2):

    import train

    train.CONFIG['epochs']       = 50
    train.CONFIG['lr_max']       = 2e-4
    train.CONFIG['batch_size']   = 24
    train.CONFIG['patch_lr']     = 96
    train.CONFIG['validate_every'] = 1
    train.CONFIG['wandb_run']    = 'v3-phase1-part1'

    # for resuming:
    # train.CONFIG['resume']       = 'wandb'
    # train.CONFIG['wandb_run_id'] = '<run_id>'

    train.main()
"""

import os
import torch
import wandb

from models.fusionsr import FusionSR, count_parameters
from models.losses import CombinedSRLoss
from training.trainer import Trainer
from data.datasets import (
    setup_ramdisk,
    make_train_dataloader,
    make_combined_dataloader,
    make_benchmark_loader,
)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG — override in notebook Cell 2 before calling main()
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # ── model architecture ──
    "in_channels": 3,
    "out_channels": 3,
    "channels": 96,
    "num_groups": 6,
    "num_rcab": 6,
    "window_size": 8,
    "num_heads": 4,
    "scale": 4,
    "ffn_expansion": 2.0,

    # ── training ──
    "mode": "general_sr",       # "general_sr" | "satellite"
    "epochs": 50,               # epochs per session
    "lr_max": 2e-4,             # peak learning rate
    "lr_min": 1e-6,             # minimum learning rate
    "sgdr_t0": 50,              # cosine period (matches session length)
    "batch_size": 24,           # per-GPU batch (×2 via DataParallel)
    "patch_lr": 96,             # LR patch size (HR = 384)
    "num_workers": 4,           # dataloader workers

    # ── optimizer (AdamW) ──
    "weight_decay": 0.01,
    "grad_clip": 1.0,           # gradient clipping max norm (0 = off)
    "warmup_epochs": 5,         # linear LR warm-up from lr_min to lr_max

    # ── validation ──
    "validate_every": 1,

    # ── loss ──
    "use_perceptual": False,    # VGG perceptual loss
    "perceptual_weight": 1.0,
    "use_gan": False,           # adversarial loss (disabled by default)
    "gan_weight": 0.005,

    # ── real-world degradation ──
    "use_degradation": False,   # Real-ESRGAN degradation pipeline

    # ── data paths (Kaggle) ──
    "div2k_base": "/kaggle/input/datasets/takihasan/div2k-dataset-for-super-resolution/Dataset",
    "flickr_base": "/kaggle/input/datasets/hliang001/flickr2k/Flickr2K",
    "use_flickr": True,         # DIV2K + Flickr2K
    "bench_base": "/kaggle/input/datasets/jesucristo/super-resolution-benchmarks",

    # ── W&B ──
    "wandb_project": "FusionSR",
    "wandb_run": "v3-phase1",
    "wandb_run_id": None,       # set to resume existing run

    # ── resume ──
    "resume": None,             # None | "wandb" | "/path/to/ckpt.pt"
    "reset_best_psnr": False,   # reset best PSNR and scheduler on resume

    # ── paths ──
    "save_dir": "/kaggle/working/checkpoints",
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\ndevice: {device}")

    # ── GPU info ──
    if device.type == "cuda":
        n_gpu = torch.cuda.device_count()
        for i in range(n_gpu):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
    else:
        n_gpu = 0

    # ── W&B ──
    if CONFIG["resume"] and CONFIG["wandb_run_id"]:
        run = wandb.init(
            project=CONFIG["wandb_project"],
            id=CONFIG["wandb_run_id"],
            resume="must",
        )
    else:
        run = wandb.init(
            project=CONFIG["wandb_project"],
            name=CONFIG["wandb_run"],
            config=CONFIG,
        )
    print(f"W&B run ID: {run.id}\n")

    # ── dataloaders ──
    div2k_base = CONFIG["div2k_base"]

    if CONFIG["use_flickr"]:
        flickr_base = CONFIG["flickr_base"]
        dst = setup_ramdisk({
            "flickr_hr": f"{flickr_base}/Flickr2K_HR",
            "flickr_lr": f"{flickr_base}/Flickr2K_LR_bicubic/X4",
        })
        train_dl = make_combined_dataloader(
            div2k_hr=f"{div2k_base}/DIV2K_train_HR",
            div2k_lr=f"{div2k_base}/DIV2K_train_LR_bicubic_X4/X4",
            flickr_hr=dst["flickr_hr"],
            flickr_lr=dst["flickr_lr"],
            patch_lr=CONFIG["patch_lr"],
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"],
        )
    else:
        dst = setup_ramdisk({
            "train_hr": f"{div2k_base}/DIV2K_train_HR",
            "train_lr": f"{div2k_base}/DIV2K_train_LR_bicubic_X4/X4",
        })
        train_dl = make_train_dataloader(
            train_hr=dst["train_hr"],
            train_lr=dst["train_lr"],
            patch_lr=CONFIG["patch_lr"],
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"],
        )

    bench_base = CONFIG["bench_base"]
    valid_dl = make_benchmark_loader(
        hr_dir=f"{bench_base}/Set5/Set5/GTmod12",
        lr_dir=f"{bench_base}/Set5/Set5/LRbicx4",
    )

    print(f"train batches: {len(train_dl)} | valid images: {len(valid_dl)}")

    # ── model ──
    model = FusionSR(
        in_channels=CONFIG["in_channels"],
        out_channels=CONFIG["out_channels"],
        channels=CONFIG["channels"],
        num_groups=CONFIG["num_groups"],
        num_rcab=CONFIG["num_rcab"],
        window_size=CONFIG["window_size"],
        num_heads=CONFIG["num_heads"],
        scale=CONFIG["scale"],
        ffn_expansion=CONFIG["ffn_expansion"],
    )

    # DataParallel for T4×2 — batch split across GPUs
    if n_gpu > 1:
        print(f"using {n_gpu} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    print(f"parameters: {count_parameters(model) / 1e6:.2f}M")

    # ── loss ──
    loss_fn = CombinedSRLoss(
        pixel_weight=1.0,
        perceptual_weight=CONFIG["perceptual_weight"],
        use_perceptual=CONFIG["use_perceptual"],
    ).to(device)

    # ── optimizer (AdamW — better for transformers than Adam) ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr_max"],
        weight_decay=CONFIG["weight_decay"],
        betas=(0.9, 0.999),
    )

    # ── optional: discriminator ──
    discriminator = None
    disc_optimizer = None
    if CONFIG["use_gan"]:
        from models.discriminator import VGGStyleDiscriminator
        discriminator = VGGStyleDiscriminator()
        if n_gpu > 1:
            discriminator = torch.nn.DataParallel(discriminator)
        discriminator = discriminator.to(device)
        disc_optimizer = torch.optim.AdamW(
            discriminator.parameters(), lr=1e-4, weight_decay=0.01,
        )
        print(f"discriminator: {count_parameters(discriminator) / 1e6:.2f}M")

    # ── optional: degradation pipeline ──
    degradation_fn = None
    if CONFIG["use_degradation"]:
        from data.degradation import RealESRGANDegradation
        degradation_fn = RealESRGANDegradation(scale=CONFIG["scale"]).to(device)
        print("real-world degradation pipeline: enabled")

    # ── trainer ──
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_dl=train_dl,
        valid_dl=valid_dl,
        config=CONFIG,
        device=device,
        save_dir=CONFIG["save_dir"],
        discriminator=discriminator,
        disc_optimizer=disc_optimizer,
        degradation_fn=degradation_fn,
    )

    # ── resume ──
    if CONFIG["resume"] == "wandb":
        ckpt_path = Trainer.download_checkpoint(CONFIG["wandb_project"])
        trainer.load_checkpoint(ckpt_path, reset_best_psnr=CONFIG["reset_best_psnr"])
    elif CONFIG["resume"]:
        trainer.load_checkpoint(CONFIG["resume"], reset_best_psnr=CONFIG["reset_best_psnr"])

    # ── train ──
    trainer.fit(
        epochs=CONFIG["epochs"],
        lr_max=CONFIG["lr_max"],
        lr_min=CONFIG["lr_min"],
        validate_every=CONFIG["validate_every"],
    )

    # ── post-training benchmark ──
    print("\n" + "=" * 60)
    print("post-training benchmark evaluation")
    print("=" * 60)
    for name, hr_sub, lr_sub in [
        ("Set5", "Set5/Set5/GTmod12", "Set5/Set5/LRbicx4"),
        ("Set14", "Set14/Set14/GTmod12", "Set14/Set14/LRbicx4"),
    ]:
        dl = make_benchmark_loader(
            hr_dir=f"{bench_base}/{hr_sub}",
            lr_dir=f"{bench_base}/{lr_sub}",
        )
        m = trainer.validate_benchmark(dl, name)
        print(f"  {name:6s} — PSNR: {m['psnr']:.2f}dB | SSIM: {m['ssim']:.4f}")
        wandb.log({
            f"benchmark/{name}/psnr": m["psnr"],
            f"benchmark/{name}/ssim": m["ssim"],
        })

    wandb.finish()
    print("\ndone.")


if __name__ == "__main__":
    main()
