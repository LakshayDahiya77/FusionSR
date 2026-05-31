"""
FusionSR-v3 trainer.

Changes from v2:
    - AdamW optimizer with weight decay
    - Gradient clipping (max_norm=1.0 by default)
    - Linear warm-up for first N epochs (avoids early instability)
    - CombinedSRLoss returns (loss, loss_dict) for component logging
    - Optional GAN discriminator update loop
    - Optional Real-ESRGAN degradation pipeline
    - SSIM in per-epoch output and W&B logging
    - Checkpoint saves discriminator state when GAN is active
"""

import os
import time
import torch
import torch.nn as nn
import wandb

from utils.metrics import psnr, ssim
from data.datasets import gpu_augment


class Trainer:
    """Training loop with checkpoint management, W&B logging, and optional GAN."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dl,
        valid_dl,
        config: dict,
        device: torch.device,
        save_dir: str = "/kaggle/working/checkpoints",
        discriminator: nn.Module = None,
        disc_optimizer: torch.optim.Optimizer = None,
        degradation_fn: nn.Module = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.config = config
        self.device = device
        self.save_dir = save_dir
        self.discriminator = discriminator
        self.disc_optimizer = disc_optimizer
        self.degradation_fn = degradation_fn

        self.scaler = torch.amp.GradScaler("cuda")
        self.grad_clip = config.get("grad_clip", 0.0)
        self.warmup_epochs = config.get("warmup_epochs", 0)

        self.best_psnr = 0.0
        self.start_epoch = 0

        # SGDR scheduler — T_0 matches session length for automatic warm restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config["sgdr_t0"],
            T_mult=1,
            eta_min=config["lr_min"],
        )

        # GAN components
        if discriminator is not None:
            self.disc_scaler = torch.amp.GradScaler("cuda")
            from models.losses import GANLoss
            self.gan_loss = GANLoss()

        os.makedirs(save_dir, exist_ok=True)

    # ── single training epoch ─────────────
    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        if self.discriminator is not None:
            self.discriminator.train()
        total_loss = 0.0
        is_satellite = self.config.get("mode") == "satellite"

        for batch in self.train_dl:
            if is_satellite:
                hr_imgs = batch.to(self.device, non_blocking=True)
                from data.datasets import generate_lr_on_gpu
                lr_imgs = generate_lr_on_gpu(hr_imgs, scale=self.config["scale"])
                lr_imgs, hr_imgs = gpu_augment(lr_imgs, hr_imgs)
            else:
                lr_imgs, hr_imgs = batch
                lr_imgs = lr_imgs.to(self.device, non_blocking=True)
                hr_imgs = hr_imgs.to(self.device, non_blocking=True)

                # optional: replace bicubic LR with degradation-generated LR
                if self.degradation_fn is not None:
                    lr_imgs = self.degradation_fn(hr_imgs)

                lr_imgs, hr_imgs = gpu_augment(lr_imgs, hr_imgs)

            # ── generator forward ──
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = self.model(lr_imgs)
                g_loss, loss_dict = self.loss_fn(pred, hr_imgs)

            # ── optional discriminator update ──
            if self.discriminator is not None:
                # D step: maximize D(real) - D(fake)
                self.disc_optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fake_logits = self.discriminator(pred.detach())
                    real_logits = self.discriminator(hr_imgs)
                    d_loss = self.gan_loss.discriminator_loss(fake_logits, real_logits)
                self.disc_scaler.scale(d_loss).backward()
                self.disc_scaler.step(self.disc_optimizer)
                self.disc_scaler.update()

                # G's adversarial loss
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    g_fake = self.discriminator(pred)
                    g_real = real_logits.detach()
                    g_gan = self.gan_loss.generator_loss(g_fake, g_real)
                g_loss = g_loss + self.config.get("gan_weight", 0.005) * g_gan

            # ── generator backward ──
            self.scaler.scale(g_loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += g_loss.item()

        return total_loss / len(self.train_dl)

    # ── checkpoint management ─────────────
    def save_checkpoint(self, epoch: int, metrics: dict, tag: str = "latest"):
        path = os.path.join(self.save_dir, f"fusionsr_{tag}.pt")

        # unwrap DataParallel — always save clean state dict
        model_state = (
            self.model.module.state_dict()
            if isinstance(self.model, nn.DataParallel)
            else self.model.state_dict()
        )

        ckpt = {
            "epoch": epoch,
            "model": model_state,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_psnr": self.best_psnr,
            "config": self.config,
            "metrics": metrics,
        }

        # save discriminator state if GAN is active
        if self.discriminator is not None:
            disc_state = (
                self.discriminator.module.state_dict()
                if isinstance(self.discriminator, nn.DataParallel)
                else self.discriminator.state_dict()
            )
            ckpt["discriminator"] = disc_state
            ckpt["disc_optimizer"] = self.disc_optimizer.state_dict()
            ckpt["disc_scaler"] = self.disc_scaler.state_dict()

        torch.save(ckpt, path)

        artifact = wandb.Artifact(
            name=f"fusionsr-{tag}",
            type="model",
            metadata={"epoch": epoch, **{k: v for k, v in metrics.items() if k != "samples"}},
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    def load_checkpoint(self, path: str, reset_best_psnr: bool = False):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # load model weights
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt["model"])

        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])

        # load discriminator if present in checkpoint and currently active
        if self.discriminator is not None and "discriminator" in ckpt:
            if isinstance(self.discriminator, nn.DataParallel):
                self.discriminator.module.load_state_dict(ckpt["discriminator"])
            else:
                self.discriminator.load_state_dict(ckpt["discriminator"])
            if "disc_optimizer" in ckpt:
                self.disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
            if "disc_scaler" in ckpt:
                self.disc_scaler.load_state_dict(ckpt["disc_scaler"])

        if reset_best_psnr:
            self.best_psnr = 0.0
            self.start_epoch = 0

            lr_max = self.config["lr_max"]
            lr_min = self.config["lr_min"]
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr_max

            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config["sgdr_t0"],
                T_mult=1,
                eta_min=lr_min,
            )
            print(f"scheduler reset | lr_max={lr_max} | lr_min={lr_min}")
        else:
            if "scheduler" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            self.best_psnr = ckpt["best_psnr"]
            self.start_epoch = ckpt["epoch"] + 1

        print(f"resumed from epoch {ckpt['epoch']} | best PSNR {self.best_psnr:.2f}dB")

    @staticmethod
    def download_checkpoint(project: str, tag: str = "latest") -> str:
        artifact = wandb.use_artifact(f"fusionsr-{tag}:latest", type="model")
        artifact_dir = artifact.download()
        return os.path.join(artifact_dir, f"fusionsr_{tag}.pt")

    # ── W&B sample logging ────────────────
    def _log_samples(self, samples: list, epoch: int):
        panels = []
        for s in samples:
            lr_up = (
                torch.nn.functional.interpolate(
                    s["lr"].unsqueeze(0), scale_factor=4,
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
            panels.append(wandb.Image(img, caption="bicubic | SR | HR"))
        wandb.log({"samples": panels}, step=epoch)

    # ── benchmark validation ──────────────
    @torch.no_grad()
    def validate_benchmark(self, benchmark_dl, name: str) -> dict:
        """Validate on benchmark dataset (Set5, Set14). Returns psnr, ssim, samples."""
        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        samples = []
        scale = self.config["scale"]

        for i, (lr_imgs, hr_imgs, fname) in enumerate(benchmark_dl):
            lr_imgs = lr_imgs.to(self.device)
            hr_imgs = hr_imgs.to(self.device).float()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = self.model(lr_imgs).float().clamp(0, 1)

            # crop to original HR size (model handles window padding)
            hr_h, hr_w = hr_imgs.shape[-2], hr_imgs.shape[-1]
            pred = pred[:, :, :hr_h, :hr_w]

            # boundary crop — standard SR evaluation
            b = scale
            pred = pred[:, :, b:-b, b:-b]
            hr_imgs = hr_imgs[:, :, b:-b, b:-b]

            total_psnr += psnr(pred, hr_imgs)
            total_ssim += ssim(pred, hr_imgs)

            if i < 5:
                samples.append({
                    "lr": lr_imgs[0].detach().cpu(),
                    "sr": pred[0].detach().cpu(),
                    "hr": hr_imgs[0].detach().cpu(),
                    "fname": fname[0],
                })

        n = len(benchmark_dl)
        torch.cuda.empty_cache()  # VRAM Optimization: prevent memory fragmentation
        return {"psnr": total_psnr / n, "ssim": total_ssim / n, "samples": samples}

    # ── main training loop ────────────────
    def fit(self, epochs: int, lr_max: float, lr_min: float, validate_every: int = 1):
        print(f"starting training for {epochs} epochs")
        print(f"SGDR T0={self.config['sgdr_t0']} | LR {lr_max} → {lr_min}")
        if self.warmup_epochs > 0 and self.start_epoch < self.warmup_epochs:
            print(f"warm-up: {self.warmup_epochs} epochs (linear ramp)")
        if self.grad_clip > 0:
            print(f"gradient clipping: max_norm={self.grad_clip}")
        print(f"validating every {validate_every} epochs")
        print("-" * 60)

        for epoch in range(self.start_epoch, self.start_epoch + epochs):

            # ── learning rate: warm-up or SGDR ──
            if epoch < self.warmup_epochs:
                # linear warm-up overrides scheduler
                frac = (epoch + 1) / self.warmup_epochs
                warmup_lr = lr_min + (lr_max - lr_min) * frac
                for pg in self.optimizer.param_groups:
                    pg["lr"] = warmup_lr
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # ── train ──
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            train_time = time.time() - t0

            log_dict = {
                "train/loss": train_loss,
                "train/lr": current_lr,
                "time/train_epoch": train_time,
                "epoch": epoch,
            }

            # ── validate ──
            if (epoch + 1) % validate_every == 0 or epoch == self.start_epoch:
                t0 = time.time()
                metrics = self.validate_benchmark(self.valid_dl, "Set5")
                val_time = time.time() - t0

                primary_psnr = metrics["psnr"]
                primary_ssim = metrics["ssim"]

                val_log = {
                    "val/psnr": primary_psnr,
                    "val/ssim": primary_ssim,
                    "time/val_epoch": val_time,
                    "time/total_epoch": train_time + val_time,
                }
                log_dict.update(val_log)

                if (epoch + 1) % 10 == 0 and "samples" in metrics:
                    self._log_samples(metrics["samples"], epoch)

                is_best = primary_psnr > self.best_psnr
                if is_best:
                    self.best_psnr = primary_psnr
                    self.save_checkpoint(epoch, metrics, tag="best")

                # ── per-epoch output (user-requested format) ──
                best_marker = " ← best" if is_best else ""
                print(
                    f"epoch {epoch:4d} | loss {train_loss:.4f} | "
                    f"PSNR {primary_psnr:.2f}dB | SSIM {primary_ssim:.4f} | "
                    f"train {train_time:.0f}s | val {val_time:.0f}s | "
                    f"LR {current_lr:.2e}{best_marker}"
                )

                self.save_checkpoint(epoch, metrics, tag="latest")
            else:
                log_dict["time/total_epoch"] = train_time
                print(
                    f"epoch {epoch:4d} | loss {train_loss:.4f} | "
                    f"train {train_time:.0f}s | LR {current_lr:.2e}"
                )

            wandb.log(log_dict, step=epoch)

        print("-" * 60)
        print(f"training complete. best PSNR: {self.best_psnr:.2f}dB")
