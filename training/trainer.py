"""
FusionSR-v4 trainer.

DDP-ready training loop with OneCycleLR, Y-channel evaluation, AMP,
and W&B logging on the main process only.
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import wandb

from utils.metrics import psnr_y, ssim_y
from data.datasets import gpu_augment, generate_lr_on_gpu


class Trainer:
    """Training loop with OneCycleLR, Y-channel eval, and DDP-safe logging."""

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
        save_dir: str = "/kaggle/working/checkpoints",
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_master = rank == 0
        self.is_distributed = world_size > 1 and dist.is_available() and dist.is_initialized()
        self.save_dir = save_dir

        self.grad_clip = config.get("grad_clip", 1.0)

        # auto-detect AMP dtype: bfloat16 on A100+ (compute capability >= 8.0)
        gpu_cap = torch.cuda.get_device_capability(device)
        if gpu_cap[0] >= 8:
            self.amp_dtype = torch.bfloat16
            # bfloat16 doesn't need loss scaling
            self.scaler = torch.amp.GradScaler("cuda", enabled=False)
            print(f"AMP: bfloat16 (GPU capability {gpu_cap[0]}.{gpu_cap[1]})")
        else:
            self.amp_dtype = torch.float16
            self.scaler = torch.amp.GradScaler("cuda")
            print(f"AMP: float16 (GPU capability {gpu_cap[0]}.{gpu_cap[1]})")

        self.best_psnr = 0.0
        self.start_epoch = config.get("start_epoch", 0)

        # OneCycleLR — created in fit() after we know steps_per_epoch
        self.scheduler = None

        os.makedirs(save_dir, exist_ok=True)

    def _all_reduce_mean(self, value: float) -> float:
        if not self.is_distributed:
            return value
        t = torch.tensor(value, device=self.device, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t / self.world_size
        return t.item()

    # ── single training epoch ─────────────
    def train_epoch(self) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        log_interval = int(self.config.get("log_interval_steps", 0) or 0)

        for step, batch in enumerate(self.train_dl, start=1):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                lr_imgs, hr_imgs = batch
                lr_imgs = lr_imgs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                hr_imgs = hr_imgs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            else:
                hr_imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
                hr_imgs = hr_imgs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                lr_imgs = generate_lr_on_gpu(hr_imgs, scale=self.config["scale"])
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

            # OneCycleLR steps per batch
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()

            if log_interval > 0 and self.is_master and step % log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  step {step:5d}/{len(self.train_dl)} | "
                    f"loss {loss.item():.4f} | lr {current_lr:.2e}"
                )

        avg_loss = total_loss / len(self.train_dl)
        return self._all_reduce_mean(avg_loss)

    def _unwrap(self):
        """Return unwrapped model for evaluation/saving (bypasses DDP hooks)."""
        return self.model.module if hasattr(self.model, "module") else self.model

    # ── benchmark validation (Y-channel) ──
    @torch.no_grad()
    def validate_benchmark(self, benchmark_dl, name: str) -> dict:
        """Validate on benchmark dataset. Returns psnr, ssim (Y-channel), samples."""
        if not self.is_master:
            return {"psnr": 0.0, "ssim": 0.0, "samples": []}
        
        model = self._unwrap()
        model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        samples = []
        scale = self.config["scale"]

        for i, (lr_imgs, hr_imgs, fname) in enumerate(benchmark_dl):
            lr_imgs = lr_imgs.to(self.device, memory_format=torch.channels_last)
            hr_imgs = hr_imgs.to(self.device, memory_format=torch.channels_last).float()

            with torch.autocast("cuda", dtype=self.amp_dtype):
                pred = model(lr_imgs).float().clamp(0, 1)

            # crop to original HR size (model handles window padding)
            hr_h, hr_w = hr_imgs.shape[-2], hr_imgs.shape[-1]
            pred = pred[:, :, :hr_h, :hr_w]

            # boundary crop — standard SR evaluation (remove `scale` pixels)
            b = scale
            pred_crop = pred[:, :, b:-b, b:-b]
            hr_crop = hr_imgs[:, :, b:-b, b:-b]

            # Y-channel metrics — matches published SwinIR/HAT numbers
            total_psnr += psnr_y(pred_crop, hr_crop)
            total_ssim += ssim_y(pred_crop, hr_crop)

            # collect samples for W&B (use uncropped for visual comparison)
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
        """Log visual comparison (bicubic | SR | HR) to W&B Media tab."""
        if not self.is_master:
            return
        panels = []
        for s in samples:
            h, w = s["hr"].shape[-2], s["hr"].shape[-1]
            lr_up = (
                torch.nn.functional.interpolate(
                    s["lr"].unsqueeze(0), size=(h, w),
                    mode="bicubic", align_corners=False,
                )
                .squeeze(0)
                .clamp(0, 1)
            )
            lr_up = lr_up[:, :h, :w]
            sr = s["sr"][:, :h, :w]
            comparison = torch.cat([lr_up, sr, s["hr"]], dim=2)
            img = comparison.permute(1, 2, 0).numpy()
            panels.append(wandb.Image(img, caption=f"{s['fname']} — bicubic | SR | HR"))
        wandb.log({"samples": panels})

    # ── checkpoint management ─────────────
    def save_checkpoint(self, epoch: int, metrics: dict, tag: str = "latest"):
        if not self.is_master:
            return
        path = os.path.join(self.save_dir, f"fusionsr_{tag}.pt")

        ckpt = {
            "epoch": epoch,
            "model": self._unwrap().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict(),
            "best_psnr": self.best_psnr,
            "config": self.config,
            "metrics": {k: v for k, v in metrics.items() if k != "samples"},
        }
        torch.save(ckpt, path)

        # upload to W&B as artifact
        artifact = wandb.Artifact(
            name=f"fusionsr-{tag}",
            type="model",
            metadata={"epoch": epoch, "psnr": metrics.get("psnr"), "ssim": metrics.get("ssim")},
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    def load_checkpoint(self, path: str):
        """Load model weights from checkpoint. Scheduler/optimizer are fresh."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._unwrap().load_state_dict(ckpt["model"])

        # always start fresh — epoch and LR come from config, not checkpoint
        self.start_epoch = self.config["start_epoch"]
        self.best_psnr = 0.0

        m = ckpt.get("metrics", {})
        if self.is_master:
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
        min_lr = self.config.get("min_lr", 1e-7)
        warmup_epochs = self.config.get("warmup_epochs", 0)
        steps_per_epoch = len(self.train_dl)
        total_steps = total_epochs * steps_per_epoch
        warmup_steps = min(warmup_epochs * steps_per_epoch, total_steps)
        min_factor = min_lr / lr_max

        for group in self.optimizer.param_groups:
            group.setdefault("initial_lr", lr_max)

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps

            if total_steps <= warmup_steps:
                return 1.0

            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            
            import math
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return cosine * (1.0 - min_factor) + min_factor

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambda,
            last_epoch=self.start_epoch * steps_per_epoch - 1,
        )

        if self.is_master:
            print(f"training: epochs {self.start_epoch}→{total_epochs - 1} "
                  f"({total_epochs - self.start_epoch} epochs)")
            print(
                f"Warmup+Cosine: warmup_epochs={warmup_epochs} | min_lr={min_lr:.1e} | "
                f"steps/epoch={steps_per_epoch} | total_steps={total_steps}"
            )
            print(f"gradient clipping: max_norm={self.grad_clip}")
            print("-" * 60)

        for epoch in range(self.start_epoch, total_epochs):
            current_lr = self.optimizer.param_groups[0]["lr"]

            if self.is_distributed and hasattr(self.train_dl, "sampler"):
                sampler = self.train_dl.sampler
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

            # ── train ──
            t0 = time.time()
            train_loss = self.train_epoch()
            train_time = time.time() - t0

            # ── validate ──
            t0 = time.time()
            metrics = self.validate_benchmark(self.valid_dl, "Set5")
            val_time = time.time() - t0

            val_psnr = metrics["psnr"]
            val_ssim = metrics["ssim"]

            # ── log to W&B ──
            if self.is_master:
                log_dict = {
                    "train/loss": train_loss,
                    "train/lr": current_lr,
                    "val/psnr_y": val_psnr,
                    "val/ssim_y": val_ssim,
                    "time/train": train_time,
                    "time/val": val_time,
                    "epoch": epoch,
                }
                wandb.log(log_dict)

            # log visual samples every epoch
            if metrics.get("samples"):
                self._log_samples(metrics["samples"])

            # ── checkpoint ──
            is_best = val_psnr > self.best_psnr
            if is_best:
                self.best_psnr = val_psnr
                self.save_checkpoint(epoch, metrics, tag="best")

            self.save_checkpoint(epoch, metrics, tag="latest")

            # ── console output ──
            if self.is_master:
                best_marker = " <- best" if is_best else ""
                print(
                    f"epoch {epoch:4d} | loss {train_loss:.4f} | "
                    f"PSNR(Y) {val_psnr:.2f}dB | SSIM(Y) {val_ssim:.4f} | "
                    f"train {train_time:.0f}s | val {val_time:.0f}s | "
                    f"LR {current_lr:.2e}{best_marker}"
                )
            
            # Prevent deadlocks by keeping DDP processes perfectly synced
            if self.is_distributed:
                dist.barrier()

        if self.is_master:
            print("-" * 60)
            print(f"training complete. best PSNR(Y): {self.best_psnr:.2f}dB")
