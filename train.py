"""
FusionSR-v4 training entry point.

Kaggle T4x2 DDP Support.

Usage (Kaggle notebook cell):

    import train

    train.CONFIG.update({
        # Data paths (set after downloading/unzipping)
        'train_hr_dirs': ['/kaggle/input/DIV2K_train_HR', '/kaggle/input/Flickr2K_HR'],
        'train_lr_dirs': ['/kaggle/input/DIV2K_train_LR_bicubic/X4', '/kaggle/input/Flickr2K_LR_bicubic/X4'],
        'val_hr_dir': '/kaggle/input/Set5/GTmod12',
        'val_lr_dir': '/kaggle/input/Set5/LRbicx4',

        # Training
        'total_epochs': 150,
        'start_epoch': 0,
        'lr_max': 3e-4,
        'batch_size': 8,
        'patch_lr': 128,

        # W&B
        'wandb_run': 'v4-kaggle-phase1',
    })

    train.main()
"""

import os
import sys
import json
import tempfile
import glob
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from models.fusionsr import FusionSR, count_parameters
from models.losses import CombinedSRLoss
from training.trainer import Trainer
from data.datasets import make_train_dl, make_benchmark_dl


def _ensure_project_on_path():
    module_dir = os.path.dirname(os.path.abspath(__file__))
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        paths = existing.split(os.pathsep)
        if module_dir not in paths:
            os.environ["PYTHONPATH"] = module_dir + os.pathsep + existing
    else:
        os.environ["PYTHONPATH"] = module_dir


_ensure_project_on_path()


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
    "log_interval_steps": 0,    # batch log interval (0 = epoch-only)

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
    "save_dir": "/kaggle/working/checkpoints",
}


def _train_worker(rank: int, world_size: int, config_file: str):
    """Worker process for DDP training."""
    with open(config_file, "r") as f:
        config = json.load(f)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    torch.backends.cudnn.benchmark = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if rank == 0:
        print(f"\nInitialized DDP with {world_size} GPUs")

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

    train_dl = make_train_dl(
        hr_dirs=config["train_hr_dirs"],
        lr_dirs=config["train_lr_dirs"] or None,
        patch_lr=config["patch_lr"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        scale=config["scale"],
        distributed=world_size > 1,
        rank=rank,
        world_size=world_size,
    )

    valid_dl = make_benchmark_dl(
        hr_dir=config["val_hr_dir"],
        lr_dir=config["val_lr_dir"],
    )

    if rank == 0:
        print(f"train batches/gpu: {len(train_dl)} | valid images: {len(valid_dl)}")

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

    if rank == 0:
        print(f"parameters: {count_parameters(model) / 1e6:.2f}M")

    model = DDP(model, device_ids=[rank], output_device=rank)

    loss_fn = CombinedSRLoss(pixel_weight=1.0, use_perceptual=False).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr_max"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999),
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_dl=train_dl,
        valid_dl=valid_dl,
        config=config,
        device=device,
        rank=rank,
        world_size=world_size,
        save_dir=config["save_dir"],
    )

    if config["resume"]:
        resume_path = config["resume"]
        ckpt_path = resume_path

        if "/" in resume_path:
            if rank == 0:
                print(f"downloading artifact: {resume_path}")
                artifact = wandb.use_artifact(resume_path, type="model")
                artifact_dir = artifact.download()
                pt_files = glob.glob(os.path.join(artifact_dir, "*.pt"))
                if not pt_files:
                    raise FileNotFoundError(
                        f"No .pt files found in artifact {resume_path}"
                    )
                ckpt_path = pt_files[0]
                with open("/tmp/fusionsr_ckpt_path.txt", "w") as f:
                    f.write(ckpt_path)

            dist.barrier()

            if rank != 0:
                with open("/tmp/fusionsr_ckpt_path.txt", "r") as f:
                    ckpt_path = f.read().strip()

        trainer.load_checkpoint(ckpt_path)

    trainer.fit()

    if rank == 0:
        wandb.finish()
        print("\ndone.")

    dist.destroy_process_group()


def main():
    """Main entry point. Launches multiprocessing for available GPUs."""
    _ensure_project_on_path()

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs found. Kaggle requires GPUs for this script.")

    print(f"Detected {world_size} GPUs. Launching DDP...")

    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(CONFIG, f)

    try:
        mp.spawn(
            _train_worker,
            args=(world_size, path),
            nprocs=world_size,
            join=True,
        )
    finally:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    main()
