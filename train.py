import os
import sys
import torch
import wandb
from kaggle_secrets import UserSecretsClient
from training.trainer import Trainer
from models.fusionsr import FusionSR
from models.losses import CharbonnierLoss
from data.datasets import make_train_dataloader, make_benchmark_loader, setup_ramdisk
from training.trainer import Trainer

# ─────────────────────────────────────────
#  Config
# ─────────────────────────────────────────
CONFIG = {
    # model
    "in_channels": 3,
    "out_channels": 3,
    "channels": 96,
    "num_groups": 6,
    "num_rcab": 6,
    "window_size": 8,
    "num_heads": 4,
    "scale": 4,
    # benchmarks
    "bench_base": "/kaggle/input/datasets/jesucristo/super-resolution-benchmarks",
    # training
    "epochs": 50,
    "lr_max": 2e-4,
    "lr_min": 1e-6,
    "sgdr_t0": 50,  # ← new: SGDR restart period
    "batch_size": 32,
    "patch_lr": 64,
    "num_workers": 4,
    "validate_every": 5,
    # data
    "div2k_base": "/kaggle/input/datasets/takihasan/div2k-dataset-for-super-resolution/Dataset",
    # wandb
    "wandb_project": "FusionSR",
    "wandb_run": "phase1-div2k-full",  # used only for fresh runs
    "wandb_run_id": None,  # ← new: set when resuming to continue same chart
    # resume
    "resume": None,  # None | 'wandb'
    # paths
    "save_dir": "/kaggle/working/checkpoints",
}


# ─────────────────────────────────────────
#  Main
# ─────────────────────────────────────────
def main():
    # ── device ──
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

    # ── RAM disk ──
    base = CONFIG["div2k_base"]
    dst = setup_ramdisk(
        {
            "train_hr": f"{base}/DIV2K_train_HR",
            "train_lr": f"{base}/DIV2K_train_LR_bicubic_X4/X4",
        }
    )

    # ── dataloaders ──
    train_dl = make_train_dataloader(
        train_hr=dst["train_hr"],
        train_lr=dst["train_lr"],
        patch_lr=CONFIG["patch_lr"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )

    # validation dataloader — Set5
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
    ).to(device)

    from models.fusionsr import count_parameters

    print(f"parameters: {count_parameters(model)/1e6:.2f}M")

    # ── loss + optimizer ──
    loss_fn = CharbonnierLoss(eps=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr_max"])

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
        trainer.load_checkpoint(ckpt_path)
    elif CONFIG["resume"]:
        trainer.load_checkpoint(CONFIG["resume"])

    # ── train ──
    trainer.fit(
        epochs=CONFIG["epochs"],
        lr_max=CONFIG["lr_max"],
        lr_min=CONFIG["lr_min"],
        validate_every=CONFIG["validate_every"],
    )

    wandb.finish()
    print("done.")


if __name__ == "__main__":
    main()
