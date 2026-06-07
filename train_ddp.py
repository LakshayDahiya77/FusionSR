"""
DDP training entry point for FusionSR-v5.
Launched via: torchrun --nproc_per_node=2 train_ddp.py --config config.json
"""
import os, json, argparse, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from models.fusionsr import FusionSR, count_parameters
from models.losses import CombinedSRLoss
from training.trainer import Trainer
from data.datasets import make_benchmark_dl, PairedSRDataset, HROnlyDataset
import wandb

def make_ddp_train_dl(hr_dirs, lr_dirs, patch_lr, batch_size, num_workers, scale):
    from torch.utils.data import ConcatDataset
    datasets = [
        PairedSRDataset(hr, lr, patch_lr=patch_lr)
        for hr, lr in zip(hr_dirs, lr_dirs)
    ]
    ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    sampler = DistributedSampler(ds, shuffle=True, drop_last=True)
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    ), sampler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # ── DDP init ──
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_master = local_rank == 0

    if config.get("allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if config.get("matmul_precision"):
        torch.set_float32_matmul_precision(config["matmul_precision"])

    # ── W&B — master only ──
    if is_master:
        wandb.login()
        wandb.init(
            entity=config["wandb_entity"],
            project=config["wandb_project"],
            name=config["wandb_run"],
            config=config,
        )

    # ── Dataloaders ──
    train_dl, sampler = make_ddp_train_dl(
        hr_dirs=config["train_hr_dirs"],
        lr_dirs=config["train_lr_dirs"],
        patch_lr=config["patch_lr"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        scale=config["scale"],
    )
    valid_dl = make_benchmark_dl(
        hr_dir=config["val_hr_dir"],
        lr_dir=config["val_lr_dir"],
    ) if is_master else None

    if is_master:
        print(f"train batches: {len(train_dl)} | valid images: {len(valid_dl)}")

    # ── Model ──
    model = FusionSR(
        channels=config["channels"],
        num_groups=config["num_groups"],
        num_heads=config["num_heads"],
        scale=config["scale"],
        ffn_expansion=config["ffn_expansion"],
        use_hfeb=config.get("use_hfeb", True),
        use_hybrid_ca=config.get("use_hybrid_ca", True),
        use_mswa=config.get("use_mswa", True),
        use_tdca=config.get("use_tdca", True),
        tdca_num_tokens=config.get("tdca_num_tokens", 64),
        tdca_interval=config.get("tdca_interval", 2),
        hf_scale_init=config.get("hf_scale_init", 0.01),
        use_checkpoint=config.get("use_checkpoint", False),
    ).to(device)

    # ── Load checkpoint BEFORE DDP wrapping so all ranks get same weights ──
    if config.get("resume"):
        ckpt_path = None
        if is_master:
            resume_path = config["resume"]
            if "/" in resume_path:
                import glob as glob_mod
                artifact = wandb.use_artifact(resume_path, type="model")
                artifact_dir = artifact.download()
                pt_files = glob_mod.glob(os.path.join(artifact_dir, "*.pt"))
                assert pt_files, f"No .pt files found in {artifact_dir}"
                ckpt_path = pt_files[0]
            else:
                ckpt_path = resume_path

        # Broadcast checkpoint path from master to all ranks
        path_list = [ckpt_path]
        dist.broadcast_object_list(path_list, src=0)
        ckpt_path = path_list[0]

        # All ranks load the checkpoint into the base model
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model"]
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        start_epoch = config.get("start_epoch", 0)
        if is_master:
            m = ckpt.get("metrics", {})
            print(f"── checkpoint loaded ──")
            print(f"  source epoch: {ckpt.get('epoch', '?')}")
            print(f"  PSNR (Y): {m.get('psnr', 'N/A')}")
            print(f"  start_epoch: {start_epoch}")
        del ckpt  # free memory

    # Wrap with DDP AFTER loading weights
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_master:
        print(f"parameters: {count_parameters(model) / 1e6:.2f}M")

    # ── Loss + Optimizer ──
    loss_fn = CombinedSRLoss(pixel_weight=1.0, use_perceptual=False).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr_max"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999),
    )

    # ── Trainer ──
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_dl=train_dl,
        valid_dl=valid_dl,
        config=config,
        device=device,
        save_dir=config["save_dir"],
        is_master=is_master,
        sampler=sampler,
    )

    trainer.fit()

    if is_master:
        wandb.finish()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
