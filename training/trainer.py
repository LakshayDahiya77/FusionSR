"""
FusionSR-v4 trainer.

Kaggle T4x2 DDP training.

Features:
    - DistributedDataParallel support
    - OneCycleLR scheduler (step per batch)
    - Y-channel PSNR/SSIM evaluation (matches published papers)
    - Auto-detect AMP dtype
    - W&B sample logging every epoch (rank 0 only)
    - Gradient clipping
    - Clean checkpoint management (rank 0 only)
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import wandb

from utils.metrics import psnr_y, ssim_y
from data.datasets import gpu_augment


class Trainer:
    """Multi-GPU DDP training loop."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dl,
        valid_dl,
        config: dict,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        save_dir: str = "/content/checkpoints",
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.save_dir = save_dir
        self.config = config

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_dl = train_dl
        self.valid_dl = valid_dl

        self.grad_clip = config.get("grad_clip", 1.0)

        # auto-detect AMP dtype
        gpu_cap = torch.cuda.get_device_capability(device)
        if gpu_cap[0] >= 8:
            self.amp_dtype = torch.bfloat16
            self.scaler = torch.amp.GradScaler("cuda", enabled=False)
            if self.rank == 0:
                print(f"AMP: bfloat16 (GPU capability {gpu_cap[0]}.{gpu_cap[1]})")
        else:
            self.amp_dtype = torch.float16
            self.scaler = torch.amp.GradScaler("cuda")
            if self.rank == 0:
                print(f"AMP: float16 (GPU capability {gpu_cap[0]}.{gpu_cap[1]})")

        # wrap model in DDP
        self.model = nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], output_device=rank
        )

        self.best_psnr = 0.0
        self.start_epoch = config.get("start_epoch", 0)
        self.scheduler = None

        if self.rank == 0:
            os.makedirs(save_dir, exist_ok=True)

    # ── single training epoch ─────────────
    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch. Returns average loss across all GPUs."""
        self.model.train()
        
        # shuffle for DDP
        if hasattr(self.train_dl.sampler, "set_epoch"):
            self.train_dl.sampler.set_epoch(epoch)

        total_loss = 0.0

        for lr_imgs, hr_imgs in self.train_dl:
            lr_imgs = lr_imgs.to(self.device, non_blocking=True)
            hr_imgs = hr_imgs.to(self.device, non_blocking=True)
            lr_imgs, hr_imgs = gpu_augment(lr_imgs, hr_imgs)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=self.amp_dtype):
                pred = self.model(lr_imgs)
                loss, loss_dict = self.loss_fn(pred, hr_imgs)

            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()

        # average loss across all GPUs
        avg_loss = torch.tensor(total_loss / len(self.train_dl), device=self.device)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss / self.world_size

        return avg_loss.item()

    # ── benchmark validation (Y-channel) ──
    @torch.no_grad()
    def validate_benchmark(self, benchmark_dl, name: str) -> dict:
        """Validate on benchmark dataset. Run ONLY on rank 0."""
        # Un-wrap model from DDP for clean inference
        model = self.model.module
        model.eval()
        
        total_psnr = 0.0
        total_ssim = 0.0
        samples = []
        scale = self.config["scale"]

        for i, (lr_imgs, hr_imgs, fname) in enumerate(benchmark_dl):
            lr_imgs = lr_imgs.to(self.device)
            hr_imgs = hr_imgs.to(self.device).float()

            with torch.autocast("cuda", dtype=self.amp_dtype):
                pred = model(lr_imgs).float().clamp(0, 1)

            # crop to original HR size
            hr_h, hr_w = hr_imgs.shape[-2], hr_imgs.shape[-1]
            pred = pred[:, :, :hr_h, :hr_w]

            # boundary crop
            b = scale
            pred_crop = pred[:, :, b:-b, b:-b]
            hr_crop = hr_imgs[:, :, b:-b, b:-b]

            # Y-channel metrics
            total_psnr += psnr_y(pred_crop, hr_crop)
            total_ssim += ssim_y(pred_crop, hr_crop)

            samples.append({
                "lr": lr_imgs[0].detach().cpu(),
                "sr": pred[0].detach().cpu(),
                "hr": hr_imgs[0].detach().cpu(),
                "fname": fname[0],
            })

        n = len(benchmark_dl)
        torch.cuda.empty_cache()
        return {"psnr": total_psnr / n, "ssim": total_ssim / n, "samples": samples}

    # ── W&B sample logging ────────────────
    def _log_samples(self, samples: list):
        """Log visual comparison to W&B (rank 0 only)."""
        panels = []
        for s in samples:
            lr_up = (
                torch.nn.functional.interpolate(
                    s["lr"].unsqueeze(0), scale_factor=self.config["scale"],
                    mode="bicubic", align_corners=False,
                )
                .squeeze(0)
                .clamp(0, 1)
            )
            h, w = s["hr"].shape[-2], s["hr"].shape[-1]
            lr_up = lr_up[:, :h, :w]
            sr = s["sr"][:, :h, :w]
            comparison = torch.cat([lr_up, sr, s["hr"]], dim=2)
            img = comparison.permute(1, 2, 0).numpy()
            panels.append(wandb.Image(img, caption=f"{s['fname']} — bicubic | SR | HR"))
        wandb.log({"samples": panels})

    # ── checkpoint management ─────────────
    def save_checkpoint(self, epoch: int, metrics: dict, tag: str = "latest"):
        """Save checkpoint (rank 0 only)."""
        path = os.path.join(self.save_dir, f"fusionsr_{tag}.pt")
        ckpt = {
            "epoch": epoch,
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict(),
            "best_psnr": self.best_psnr,
            "config": self.config,
            "metrics": {k: v for k, v in metrics.items() if k != "samples"},
        }
        torch.save(ckpt, path)

        artifact = wandb.Artifact(
            name=f"fusionsr-{tag}",
            type="model",
            metadata={"epoch": epoch, "psnr": metrics.get("psnr"), "ssim": metrics.get("ssim")},
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    def load_checkpoint(self, path: str):
        """Load model weights (all ranks)."""
        # map to specific GPU to avoid VRAM spikes
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.module.load_state_dict(ckpt["model"])

        self.start_epoch = self.config["start_epoch"]
        self.best_psnr = 0.0

        if self.rank == 0:
            m = ckpt.get("metrics", {})
            print(f"── checkpoint loaded ──")
            print(f"  source epoch: {ckpt.get('epoch', '?')}")
            print(f"  PSNR (Y): {m.get('psnr', 'N/A')}")
            print(f"  SSIM (Y): {m.get('ssim', 'N/A')}")
            print(f"── training config ──")
            print(f"  start_epoch: {self.start_epoch}")
            print(f"  lr_max: {self.config['lr_max']}")

    # ── main training loop ────────────────
    def fit(self):
        total_epochs = self.config["total_epochs"]
        lr_max = self.config["lr_max"]
        steps_per_epoch = len(self.train_dl)
        total_steps = total_epochs * steps_per_epoch

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr_max,
            total_steps=total_steps,
            pct_start=0.05,
            anneal_strategy="cos",
            div_factor=25,
            final_div_factor=1e4,
        )

        if self.start_epoch > 0:
            steps_to_skip = self.start_epoch * steps_per_epoch
            if self.rank == 0:
                print(f"fast-forwarding scheduler by {steps_to_skip} steps...")
            for _ in range(steps_to_skip):
                self.scheduler.step()

        if self.rank == 0:
            print(f"training: epochs {self.start_epoch}→{total_epochs - 1}")
            print(f"OneCycleLR: max_lr={lr_max:.1e} | steps/epoch={steps_per_epoch}")
            print("-" * 60)

        for epoch in range(self.start_epoch, total_epochs):
            current_lr = self.optimizer.param_groups[0]["lr"]

            # ── train ──
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            t1 = time.time()

            # ── log & validate (rank 0 only) ──
            if self.rank == 0:
                metrics = {"train/loss": train_loss, "train/lr": current_lr}
                msg = f"[{epoch:03d}/{total_epochs-1:03d}] loss: {train_loss:.4f} | lr: {current_lr:.1e} | time: {t1-t0:.1f}s"

                # Validation
                m = self.validate_benchmark(self.valid_dl, "Set5")
                msg += f" | psnr: {m['psnr']:.2f} | ssim: {m['ssim']:.4f}"

                metrics["val/Set5_psnr_y"] = m["psnr"]
                metrics["val/Set5_ssim_y"] = m["ssim"]

                # Log to wandb
                wandb.log(metrics)
                self._log_samples(m["samples"])

                # Checkpoints
                self.save_checkpoint(epoch, metrics, tag="latest")
                if m["psnr"] > self.best_psnr:
                    self.best_psnr = m["psnr"]
                    self.save_checkpoint(epoch, metrics, tag="best")
                    msg += " (best)"

                print(msg)
            
            # Sync all GPUs before next epoch
            dist.barrier()
