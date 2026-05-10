import os
import torch
import wandb
from kaggle_secrets import UserSecretsClient
from training.trainer import Trainer
from models.fusionsr import FusionSR, count_parameters
from models.losses import CharbonnierLoss
from data.datasets import (
    make_train_dataloader,
    make_combined_dataloader,
    make_satellite_hr_dataloader,
    make_benchmark_loader,
    setup_ramdisk,
)

# ─────────────────────────────────────────
#  Config
# ─────────────────────────────────────────
CONFIG = {
    # ── model ──
    "in_channels": 3,
    "out_channels": 3,
    "channels": 96,
    "num_groups": 6,
    "num_rcab": 6,
    "window_size": 8,
    "num_heads": 4,
    "scale": 4,
    # ── training mode ──
    # "general_sr"  → pretrain on DIV2K / DIV2K+Flickr2K
    # "satellite"   → fine-tune on satellite dataset (DIOR)
    "mode": "general_sr",
    # ── general SR data ──
    "div2k_base": "/kaggle/input/datasets/takihasan/div2k-dataset-for-super-resolution/Dataset",
    "flickr_base": "/kaggle/input/datasets/hliang001/flickr2k/Flickr2K",
    "use_flickr": True,  # True = DIV2K + Flickr2K, False = DIV2K only
    # ── satellite data (DIOR) ──
    "dior_base": "/kaggle/input/datasets/redzapdos123/dior-r-dataset-yolov11-obb-format/YOLODIOR-R",
    "patch_hr": 256,  # HR patch size for satellite (LR = 256//4 = 64)
    # ── benchmarks ──
    "bench_base": "/kaggle/input/datasets/jesucristo/super-resolution-benchmarks",
    # ── general SR training hyperparams ──
    "epochs": 50,
    "lr_max": 2e-4,
    "lr_min": 1e-6,
    "sgdr_t0": 50,
    "batch_size": 32,
    "patch_lr": 64,
    "num_workers": 4,
    "validate_every": 1,
    # ── satellite fine-tune hyperparams ──
    "sat_lr_max": 5e-5,  # lower LR — preserve pretrained weights
    "sat_lr_min": 1e-7,
    "sat_sgdr_t0": 50,
    "sat_batch": 16,
    # ── wandb ──
    "wandb_project": "FusionSR",
    "wandb_run": "phase1-div2k-flickr",  # name for fresh runs
    "wandb_run_id": None,  # set to resume existing run e.g. "gvolrfd3"
    # ── resume ──
    "resume": None,  # None | "wandb" | "/path/to/checkpoint.pt"
    "reset_best_psnr": False,  # set True when switching general SR → satellite
    # ── paths ──
    "save_dir": "/kaggle/working/checkpoints",
}


# ─────────────────────────────────────────
#  Main
# ─────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

    # ── W&B ──
    secrets = UserSecretsClient()
    wandb.login(key=secrets.get_secret("WANDB_API_KEY"))

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
    print(f"W&B run ID: {run.id}")

    # ── dataloaders ──
    if CONFIG["mode"] == "general_sr":

        div2k_base = CONFIG["div2k_base"]
        flickr_base = CONFIG["flickr_base"]

        if CONFIG["use_flickr"]:
            # Flickr2K in RAM disk, DIV2K from SSD
            dst = setup_ramdisk(
                {
                    "flickr_hr": f"{flickr_base}/Flickr2K_HR",
                    "flickr_lr": f"{flickr_base}/Flickr2K_LR_bicubic/X4",
                }
            )
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
            # DIV2K only
            dst = setup_ramdisk(
                {
                    "train_hr": f"{div2k_base}/DIV2K_train_HR",
                    "train_lr": f"{div2k_base}/DIV2K_train_LR_bicubic_X4/X4",
                }
            )
            train_dl = make_train_dataloader(
                train_hr=dst["train_hr"],
                train_lr=dst["train_lr"],
                patch_lr=CONFIG["patch_lr"],
                batch_size=CONFIG["batch_size"],
                num_workers=CONFIG["num_workers"],
            )

        # validation — Set5
        bench_base = CONFIG["bench_base"]
        valid_dl = make_benchmark_loader(
            hr_dir=f"{bench_base}/Set5/Set5/GTmod12",
            lr_dir=f"{bench_base}/Set5/Set5/LRbicx4",
        )

        lr_max = CONFIG["lr_max"]
        lr_min = CONFIG["lr_min"]

    elif CONFIG["mode"] == "satellite":

        dior_base = CONFIG["dior_base"]

        train_dl = make_satellite_hr_dataloader(
            hr_dir=f"{dior_base}/train/images",
            patch_hr=CONFIG["patch_hr"],
            scale=CONFIG["scale"],
            batch_size=CONFIG["sat_batch"],
            num_workers=CONFIG["num_workers"],
            training=True,
        )

        valid_dl = make_satellite_hr_dataloader(
            hr_dir=f"{dior_base}/val/images",
            patch_hr=CONFIG["patch_hr"],
            scale=CONFIG["scale"],
            batch_size=1,
            num_workers=2,
            training=False,
        )

        lr_max = CONFIG["sat_lr_max"]
        lr_min = CONFIG["sat_lr_min"]
        CONFIG["sgdr_t0"] = CONFIG["sat_sgdr_t0"]

    else:
        raise ValueError(f"unknown mode: {CONFIG['mode']}")

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
    )

    if torch.cuda.device_count() > 1:
        print(f"using {torch.cuda.device_count()} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    print(f"parameters: {count_parameters(model)/1e6:.2f}M")

    # ── loss + optimizer ──
    loss_fn = CharbonnierLoss(eps=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_max)

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
    )

    # ── resume ──
    if CONFIG["resume"] == "wandb":
        ckpt_path = Trainer.download_checkpoint(CONFIG["wandb_project"])
        trainer.load_checkpoint(ckpt_path, reset_best_psnr=CONFIG["reset_best_psnr"])
    elif CONFIG["resume"]:
        trainer.load_checkpoint(
            CONFIG["resume"], reset_best_psnr=CONFIG["reset_best_psnr"]
        )

    # ── train ──
    trainer.fit(
        epochs=CONFIG["epochs"],
        lr_max=lr_max,
        lr_min=lr_min,
        validate_every=CONFIG["validate_every"],
    )

    # ── post-training benchmark eval ──
    if CONFIG["mode"] == "general_sr":
        print("\nrunning benchmark evaluation...")
        bench_base = CONFIG["bench_base"]
        for name, hr_sub, lr_sub in [
            ("Set5", "Set5/Set5/GTmod12", "Set5/Set5/LRbicx4"),
            ("Set14", "Set14/Set14/GTmod12", "Set14/Set14/LRbicx4"),
        ]:
            dl = make_benchmark_loader(
                hr_dir=f"{bench_base}/{hr_sub}",
                lr_dir=f"{bench_base}/{lr_sub}",
            )
            m = trainer.validate_benchmark(dl, name)
            print(f"{name:6s} — PSNR: {m['psnr']:.2f}dB | SSIM: {m['ssim']:.4f}")
            wandb.log(
                {
                    f"benchmark/{name}/psnr": m["psnr"],
                    f"benchmark/{name}/ssim": m["ssim"],
                }
            )

    elif CONFIG["mode"] == "satellite":
        print("\nrunning final satellite validation...")
        m = trainer.validate_satellite(valid_dl, "DIOR-val")
        print(
            f"DIOR val — RGB PSNR: {m['psnr_rgb']:.2f}dB | "
            f"Y PSNR: {m['psnr_y']:.2f}dB | "
            f"SSIM Y: {m['ssim_y']:.4f}"
        )
        wandb.log(
            {
                "final/psnr_rgb": m["psnr_rgb"],
                "final/psnr_y": m["psnr_y"],
                "final/ssim_y": m["ssim_y"],
            }
        )

    wandb.finish()
    print("done.")


if __name__ == "__main__":
    main()
