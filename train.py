"""
FusionSR-v3 training entry point.

Usage (Kaggle notebook Cell 2):

    import train

    train.CONFIG['epochs']       = 50
    train.CONFIG['lr_max']       = 5e-5
    train.CONFIG['batch_size']   = 24
    train.CONFIG['wandb_run']    = 'v3-phase1'

    # for resuming:
    # train.CONFIG['resume']       = 'wandb'
    # train.CONFIG['wandb_run_id'] = '<run_id>'

    train.main()
"""

import os
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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
    "sgdr_t0": 250,             # cosine period — set to total planned epochs
    "batch_size": 24,           # per-GPU batch size
    "patch_lr": 96,             # LR patch size (HR = 384)
    "num_workers": 4,           # dataloader workers per GPU
    "amp_dtype": "float16",     # "float16" for T4/V100, "bfloat16" for A100+

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
    "use_ramdisk": False,       # copy datasets to /dev/shm (consumes ~20GB RAM)
    "div2k_base": "/kaggle/input/datasets/takihasan/div2k-dataset-for-super-resolution/Dataset",
    "flickr_base": "/kaggle/input/datasets/hliang001/flickr2k/Flickr2K",
    "use_flickr": True,         # DIV2K + Flickr2K
    "bench_base": "/kaggle/input/datasets/jesucristo/super-resolution-benchmarks",

    # ── W&B ──
    "wandb_project": "FusionSR",
    "wandb_run": "v3-phase1",
    "wandb_run_id": None,       # set to resume existing run

    # ── resume ──
    "resume": None,             # W&B artifact name (e.g. "user/project/fusionsr-best:v260") or local path
    "start_epoch": 0,           # epoch number to start from

    # ── paths ──
    "save_dir": "/kaggle/working/checkpoints",
}

_CONFIG_PATH = "/tmp/fusionsr_config.json"


def _train_worker(rank: int, world_size: int):
    """Training worker — called once per GPU process.
    
    Reads config from a JSON file written by main(). This avoids the problem
    where mp.spawn re-imports the module and loses notebook CONFIG overrides.
    """
    with open(_CONFIG_PATH) as f:
        config = json.load(f)

    is_main = (rank == 0)
    use_ddp = (world_size > 1)

    # ── DDP process group ──
    if use_ddp:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}")
    torch.backends.cudnn.benchmark = True

    try:
        _run_training(rank, world_size, device, is_main, use_ddp, config)
    finally:
        if use_ddp:
            dist.destroy_process_group()


def _run_training(rank, world_size, device, is_main, use_ddp, config):
    """Core training logic — isolated for clean DDP cleanup."""

    # ── GPU info ──
    if is_main:
        print(f"\ndevice: {device}")
        n_total = torch.cuda.device_count()
        for i in range(n_total):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
        if use_ddp:
            print(f"using {world_size} GPUs via DistributedDataParallel")

    # ── W&B (rank 0 only) ──
    if is_main:
        if config["resume"] and config["wandb_run_id"]:
            run = wandb.init(
                project=config["wandb_project"],
                id=config["wandb_run_id"],
                resume="must",
            )
        else:
            run = wandb.init(
                project=config["wandb_project"],
                name=config["wandb_run"],
                config=config,
            )
        print(f"W&B run ID: {run.id}\n")

    # ── dataloaders ──
    div2k_base = config["div2k_base"]

    if config["use_flickr"]:
        flickr_base = config["flickr_base"]
        if config["use_ramdisk"]:
            dst = setup_ramdisk({
                "flickr_hr": f"{flickr_base}/Flickr2K_HR",
                "flickr_lr": f"{flickr_base}/Flickr2K_LR_bicubic/X4",
            })
            flickr_hr_path = dst["flickr_hr"]
            flickr_lr_path = dst["flickr_lr"]
        else:
            flickr_hr_path = f"{flickr_base}/Flickr2K_HR"
            flickr_lr_path = f"{flickr_base}/Flickr2K_LR_bicubic/X4"

        train_dl = make_combined_dataloader(
            div2k_hr=f"{div2k_base}/DIV2K_train_HR",
            div2k_lr=f"{div2k_base}/DIV2K_train_LR_bicubic_X4/X4",
            flickr_hr=flickr_hr_path,
            flickr_lr=flickr_lr_path,
            patch_lr=config["patch_lr"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            distributed=use_ddp,
        )
    else:
        if config["use_ramdisk"]:
            dst = setup_ramdisk({
                "train_hr": f"{div2k_base}/DIV2K_train_HR",
                "train_lr": f"{div2k_base}/DIV2K_train_LR_bicubic_X4/X4",
            })
            train_hr_path = dst["train_hr"]
            train_lr_path = dst["train_lr"]
        else:
            train_hr_path = f"{div2k_base}/DIV2K_train_HR"
            train_lr_path = f"{div2k_base}/DIV2K_train_LR_bicubic_X4/X4"

        train_dl = make_train_dataloader(
            train_hr=train_hr_path,
            train_lr=train_lr_path,
            patch_lr=config["patch_lr"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            distributed=use_ddp,
        )

    bench_base = config["bench_base"]
    valid_dl = make_benchmark_loader(
        hr_dir=f"{bench_base}/Set5/Set5/GTmod12",
        lr_dir=f"{bench_base}/Set5/Set5/LRbicx4",
    )

    if is_main:
        print(f"train batches: {len(train_dl)} | valid images: {len(valid_dl)}")
        print(f"batch_size: {config['batch_size']} | amp: {config['amp_dtype']}")

    # ── model ──
    model = FusionSR(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        channels=config["channels"],
        num_groups=config["num_groups"],
        num_rcab=config["num_rcab"],
        window_size=config["window_size"],
        num_heads=config["num_heads"],
        scale=config["scale"],
        ffn_expansion=config["ffn_expansion"],
    )
    model = model.to(device)

    # DDP for multi-GPU (torch.compile skipped — T4 inductor is too slow)
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    if is_main:
        print(f"parameters: {count_parameters(model) / 1e6:.2f}M")

    # ── loss ──
    loss_fn = CombinedSRLoss(
        pixel_weight=1.0,
        perceptual_weight=config["perceptual_weight"],
        use_perceptual=config["use_perceptual"],
    ).to(device)

    # ── optimizer (AdamW — better for transformers than Adam) ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr_max"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999),
    )

    # ── optional: discriminator ──
    discriminator = None
    disc_optimizer = None
    if config["use_gan"]:
        from models.discriminator import VGGStyleDiscriminator
        discriminator = VGGStyleDiscriminator().to(device)
        if use_ddp:
            discriminator = torch.nn.parallel.DistributedDataParallel(
                discriminator, device_ids=[rank]
            )
        disc_optimizer = torch.optim.AdamW(
            discriminator.parameters(), lr=1e-4, weight_decay=0.01,
        )
        if is_main:
            print(f"discriminator: {count_parameters(discriminator) / 1e6:.2f}M")

    # ── optional: degradation pipeline ──
    degradation_fn = None
    if config["use_degradation"]:
        from data.degradation import RealESRGANDegradation
        degradation_fn = RealESRGANDegradation(scale=config["scale"]).to(device)
        if is_main:
            print("real-world degradation pipeline: enabled")

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
        discriminator=discriminator,
        disc_optimizer=disc_optimizer,
        degradation_fn=degradation_fn,
        rank=rank,
    )

    # ── resume ──
    if config["resume"]:
        resume_path = config["resume"]

        # download from W&B if it looks like an artifact name
        if "/" in resume_path:
            if is_main:
                print(f"downloading artifact: {resume_path}")
                artifact = wandb.use_artifact(resume_path, type="model")
                artifact_dir = artifact.download()
                # find the .pt file inside
                import glob
                pt_files = glob.glob(os.path.join(artifact_dir, "*.pt"))
                ckpt_path = pt_files[0]
            else:
                ckpt_path = ""

            if use_ddp:
                path_list = [ckpt_path]
                dist.broadcast_object_list(path_list, src=0)
                ckpt_path = path_list[0]
        else:
            ckpt_path = resume_path

        trainer.load_checkpoint(ckpt_path)

    if use_ddp:
        dist.barrier()

    # ── train ──
    trainer.fit(
        epochs=config["epochs"],
        lr_max=config["lr_max"],
        lr_min=config["lr_min"],
        validate_every=config["validate_every"],
    )

    # ── post-training benchmark (rank 0 only) ──
    if is_main:
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


def main():
    """Entry point — spawns DDP workers for multi-GPU, or runs directly for single GPU."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Save CONFIG to temp file — mp.spawn re-imports the module,
    # so notebook overrides to CONFIG would be lost without this
    with open(_CONFIG_PATH, "w") as f:
        json.dump(CONFIG, f)

    if n_gpu > 1:
        mp.spawn(_train_worker, args=(n_gpu,), nprocs=n_gpu, join=True)
    else:
        _train_worker(0, 1)


if __name__ == "__main__":
    main()
