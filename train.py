"""
FusionSR-v4 training entry point.

Kaggle T4x2 DDP Support.

Usage (Kaggle notebook cell):

    import train
    train.CONFIG.update({...})
    train.main()
"""

import os
import json
import tempfile
import glob
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
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
    "use_checkpoint": True,     # saves massive VRAM at the cost of ~20% compute time

    # ── training ──
    "total_epochs": 150,        
    "start_epoch": 0,           
    "lr_max": 3e-4,             
    "batch_size": 8,            # per-GPU batch size
    "patch_lr": 128,            
    "num_workers": 2,           # per-GPU workers
    "weight_decay": 0.01,       
    "grad_clip": 1.0,           

    # ── data paths ──
    "train_hr_dirs": [],        
    "train_lr_dirs": [],        
    "val_hr_dir": "",           
    "val_lr_dir": "",           

    # ── W&B ──
    "wandb_entity": "lakshay_dahiya77",
    "wandb_project": "FusionSR-v4",
    "wandb_run": "v4-kaggle-phase1",
    "wandb_run_id": None,       

    # ── resume ──
    "resume": None,             

    # ── paths ──
    "save_dir": "/kaggle/working/checkpoints",
}


def _train_worker(rank: int, world_size: int, config_file: str):
    """Worker process for DDP training."""
    # load config from temp file (preserves notebook overrides)
    with open(config_file, "r") as f:
        config = json.load(f)

    # Initialize process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    torch.backends.cudnn.benchmark = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if rank == 0:
        print(f"\nInitialized DDP with {world_size} GPUs")
        
        # W&B init only on rank 0
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
    train_dl = make_train_dl(
        hr_dirs=config["train_hr_dirs"],
        lr_dirs=config["train_lr_dirs"] or None,
        patch_lr=config["patch_lr"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        scale=config["scale"],
        distributed=True,
    )

    valid_dl = make_benchmark_dl(
        hr_dir=config["val_hr_dir"],
        lr_dir=config["val_lr_dir"],
    )

    if rank == 0:
        print(f"train batches/gpu: {len(train_dl)} | valid images: {len(valid_dl)}")

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

    if rank == 0:
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
        rank=rank,
        world_size=world_size,
        save_dir=config["save_dir"],
    )

    # ── resume ──
    if config["resume"]:
        resume_path = config["resume"]
        ckpt_path = resume_path

        if "/" in resume_path:
            # W&B artifact
            if rank == 0:
                print(f"downloading artifact: {resume_path}")
                artifact = wandb.use_artifact(resume_path, type="model")
                artifact_dir = artifact.download()
                pt_files = glob.glob(os.path.join(artifact_dir, "*.pt"))
                ckpt_path = pt_files[0]
                # save path for other ranks
                with open("/tmp/fusionsr_ckpt_path.txt", "w") as f:
                    f.write(ckpt_path)
            
            # Wait for rank 0 to finish downloading and saving path
            dist.barrier()
            
            if rank != 0:
                with open("/tmp/fusionsr_ckpt_path.txt", "r") as f:
                    ckpt_path = f.read().strip()
                    
        trainer.load_checkpoint(ckpt_path)

    # ── train ──
    trainer.fit()

    if rank == 0:
        wandb.finish()
        print("\ndone.")
        
    dist.destroy_process_group()


def main():
    """Main entry point. Launches multiprocessing for available GPUs."""
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs found. Kaggle requires GPUs for this script.")
        
    print(f"Detected {world_size} GPUs. Launching DDP...")

    # save config to temp JSON so spawned processes can read overrides
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
