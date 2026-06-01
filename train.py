"""
FusionSR-v4 training entry point.

Usage (Colab notebook cell):

    import train

    train.CONFIG.update({
        # Data paths (set after downloading/unzipping)
        'train_hr_dirs': ['/content/DIV2K_train_HR', '/content/Flickr2K_HR'],
        'train_lr_dirs': ['/content/DIV2K_train_LR_bicubic/X4', '/content/Flickr2K_LR_bicubic/X4'],
        'val_hr_dir': '/content/Set5/GTmod12',
        'val_lr_dir': '/content/Set5/LRbicx4',

        # Training
        'total_epochs': 150,
        'start_epoch': 0,
        'lr_max': 3e-4,
        'batch_size': 16,
        'patch_lr': 128,

        # W&B
        'wandb_run': 'v4-phase1',
    })

    train.main()
"""

import os
import glob
import torch
import wandb

from models.fusionsr import FusionSR, count_parameters
from models.losses import CombinedSRLoss
from training.trainer import Trainer
from data.datasets import make_train_dl, make_benchmark_dl


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG — override in notebook cell before calling main()
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # ── model architecture ──
    "channels": 180,
    "num_groups": 6,
    "num_rcab": 6,
    "window_size": 16,
    "num_heads": 6,
    "scale": 4,
    "ffn_expansion": 2.0,
    "oca_overlap": 4,

    # ── training ──
    "total_epochs": 150,        # OneCycleLR schedule length
    "start_epoch": 0,           # resume from this epoch
    "lr_max": 3e-4,             # OneCycleLR peak LR
    "batch_size": 16,           # adjust based on GPU VRAM
    "patch_lr": 128,            # LR patch size (HR = 512)
    "num_workers": 4,           # dataloader workers
    "weight_decay": 0.01,       # AdamW weight decay
    "grad_clip": 1.0,           # gradient clipping max norm
    "use_checkpoint": True,     # saves massive VRAM at the cost of ~20% compute time

    # ── data paths (set in notebook cell) ──
    "train_hr_dirs": [],        # list of HR image directories
    "train_lr_dirs": [],        # list of LR directories (empty = generate on-the-fly)
    "val_hr_dir": "",           # Set5 GTmod12 path
    "val_lr_dir": "",           # Set5 LRbicx4 path

    # ── W&B ──
    "wandb_entity": "lakshay_dahiya77",
    "wandb_project": "FusionSR-v4",
    "wandb_run": "v4-phase1",
    "wandb_run_id": None,       # set to resume same W&B run

    # ── resume ──
    "resume": None,             # W&B artifact name or local .pt path

    # ── paths ──
    "save_dir": "/content/checkpoints",
}


def main():
    """Single-GPU training entry point."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    config = CONFIG.copy()

    # ── GPU info ──
    print(f"\ndevice: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name} ({props.total_memory / 1024**3:.1f}GB)")

    # ── W&B ──
    wandb.login()

    wandb_kwargs = {
        "entity": config["wandb_entity"],
        "project": config["wandb_project"],
        "config": config,
    }
    if config["wandb_run_id"]:
        wandb_kwargs["id"] = config["wandb_run_id"]
        wandb_kwargs["resume"] = "must"
    else:
        wandb_kwargs["name"] = config["wandb_run"]

    run = wandb.init(**wandb_kwargs)
    print(f"W&B run ID: {run.id}\n")

    # ── dataloaders ──
    assert config["train_hr_dirs"], "Set CONFIG['train_hr_dirs'] to a list of HR directories"
    assert config["val_hr_dir"], "Set CONFIG['val_hr_dir'] to Set5 GTmod12 path"
    assert config["val_lr_dir"], "Set CONFIG['val_lr_dir'] to Set5 LRbicx4 path"

    train_dl = make_train_dl(
        hr_dirs=config["train_hr_dirs"],
        lr_dirs=config["train_lr_dirs"] or None,
        patch_lr=config["patch_lr"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        scale=config["scale"],
    )

    valid_dl = make_benchmark_dl(
        hr_dir=config["val_hr_dir"],
        lr_dir=config["val_lr_dir"],
    )

    print(f"train batches: {len(train_dl)} | valid images: {len(valid_dl)}")

    # ── model ──
    model = FusionSR(
        channels=config["channels"],
        num_groups=config["num_groups"],
        num_rcab=config["num_rcab"],
        window_size=config["window_size"],
        num_heads=config["num_heads"],
        scale=config["scale"],
        ffn_expansion=config["ffn_expansion"],
        oca_overlap=config["oca_overlap"],
        use_checkpoint=config.get("use_checkpoint", False),
    ).to(device)

    print(f"parameters: {count_parameters(model) / 1e6:.2f}M")

    # ── loss ──
    loss_fn = CombinedSRLoss(pixel_weight=1.0, use_perceptual=False).to(device)

    # ── optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr_max"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999),
    )

    # ── trainer ──
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_dl=train_dl,
        valid_dl=valid_dl,
        config=config,
        device=device,
        save_dir=config["save_dir"],
    )

    # ── resume ──
    if config["resume"]:
        resume_path = config["resume"]

        if "/" in resume_path:
            # W&B artifact — download
            print(f"downloading artifact: {resume_path}")
            artifact = wandb.use_artifact(resume_path, type="model")
            artifact_dir = artifact.download()
            pt_files = glob.glob(os.path.join(artifact_dir, "*.pt"))
            assert pt_files, f"No .pt files found in artifact {resume_path}"
            ckpt_path = pt_files[0]
        else:
            ckpt_path = resume_path

        trainer.load_checkpoint(ckpt_path)

    # ── train ──
    trainer.fit()

    # ── post-training benchmark ──
    print("\n" + "=" * 60)
    print("post-training benchmark")
    print("=" * 60)
    m = trainer.validate_benchmark(valid_dl, "Set5")
    print(f"  Set5 — PSNR(Y): {m['psnr']:.2f}dB | SSIM(Y): {m['ssim']:.4f}")

    wandb.finish()
    print("\ndone.")


if __name__ == "__main__":
    main()
