import os
import time
import math
import torch
import torch.nn as nn
from utils.metrics import psnr, ssim
import wandb


# ─────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────
class Trainer:
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
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.config = config
        self.device = device
        self.save_dir = save_dir
        self.scaler = torch.amp.GradScaler("cuda")

        self.best_psnr = 0.0
        self.start_epoch = 0

        # SGDR scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config["sgdr_t0"],
            T_mult=1,
            eta_min=config["lr_min"],
        )

        os.makedirs(save_dir, exist_ok=True)

    # ── training ──────────────────────────
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        is_satellite = self.config.get("mode") == "satellite"

        for batch in self.train_dl:
            if is_satellite:
                hr_imgs = batch.to(self.device, non_blocking=True)
                from data.datasets import generate_lr_on_gpu, gpu_augment

                lr_imgs = generate_lr_on_gpu(hr_imgs, scale=self.config["scale"])
                lr_imgs, hr_imgs = gpu_augment(lr_imgs, hr_imgs)
            else:
                lr_imgs, hr_imgs = batch
                lr_imgs = lr_imgs.to(self.device, non_blocking=True)
                hr_imgs = hr_imgs.to(self.device, non_blocking=True)
                from data.datasets import gpu_augment

                lr_imgs, hr_imgs = gpu_augment(lr_imgs, hr_imgs)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = self.model(lr_imgs)
                loss = self.loss_fn(pred, hr_imgs)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()

        return total_loss / len(self.train_dl)

    # ── checkpoint ────────────────────────
    def save_checkpoint(self, epoch: int, metrics: dict, tag: str = "latest"):
        path = os.path.join(self.save_dir, f"fusionsr_{tag}.pt")

        # unwrap DataParallel if present — always save clean state dict
        model_state = (
            self.model.module.state_dict()
            if isinstance(self.model, nn.DataParallel)
            else self.model.state_dict()
        )

        torch.save(
            {
                "epoch": epoch,
                "model": model_state,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),  # ← saves LR position
                "scaler": self.scaler.state_dict(),
                "best_psnr": self.best_psnr,
                "config": self.config,
                "metrics": metrics,
            },
            path,
        )

        artifact = wandb.Artifact(
            name=f"fusionsr-{tag}", type="model", metadata={"epoch": epoch, **metrics}
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    def load_checkpoint(self, path: str, reset_best_psnr: bool = False):
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])

        if reset_best_psnr:
            # new phase — reset scheduler, epoch counter, best PSNR
            self.best_psnr = 0.0
            self.start_epoch = 0
            # reinitialize optimizer LR to match new config
            for pg in self.optimizer.param_groups:
                pg["lr"] = (
                    self.config["sat_lr_max"]
                    if self.config["mode"] == "satellite"
                    else self.config["lr_max"]
                )
            # reinitialize scheduler from scratch
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config["sgdr_t0"],
                T_mult=1,
                eta_min=self.config["lr_min"],
            )
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

    # ── W&B image logging ─────────────────
    def _log_samples(self, samples: list, epoch: int):
        panels = []
        for s in samples:
            lr_up = (
                torch.nn.functional.interpolate(
                    s["lr"].unsqueeze(0),
                    scale_factor=4,
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze(0)
                .clamp(0, 1)
            )

            # ensure all three have matching spatial size before concat
            h, w = s["hr"].shape[-2], s["hr"].shape[-1]
            lr_up = lr_up[:, :h, :w]
            sr = s["sr"][:, :h, :w]

            comparison = torch.cat([lr_up, sr, s["hr"]], dim=2)
            img = comparison.permute(1, 2, 0).numpy()
            panels.append(wandb.Image(img, caption="bicubic | SR | HR"))

        wandb.log({"samples": panels}, step=epoch)

    # ── validation on benchmark datasets ──
    @torch.no_grad()
    def validate_benchmark(self, benchmark_dl, name: str) -> dict:
        """
        Full image validation on benchmark datasets (Set5, Set14).
        Pads to window size, runs full image, crops boundary before metrics.
        Standard SR evaluation methodology.
        """
        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        samples = []
        window_size = self.config["window_size"]
        scale = self.config["scale"]

        for i, (lr_imgs, hr_imgs, fname) in enumerate(benchmark_dl):
            lr_imgs = lr_imgs.to(self.device)
            hr_imgs = hr_imgs.to(self.device).float()

            # pad to window size
            _, _, h, w = lr_imgs.shape
            pad_h = (window_size - h % window_size) % window_size
            pad_w = (window_size - w % window_size) % window_size
            if pad_h > 0 or pad_w > 0:
                lr_imgs = torch.nn.functional.pad(
                    lr_imgs, (0, pad_w, 0, pad_h), mode="reflect"
                )

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = self.model(lr_imgs).float().clamp(0, 1)

            # crop to original HR size
            hr_h, hr_w = hr_imgs.shape[-2], hr_imgs.shape[-1]
            pred = pred[:, :, :hr_h, :hr_w]

            # boundary crop — standard SR evaluation
            b = scale
            pred = pred[:, :, b:-b, b:-b]
            hr_imgs = hr_imgs[:, :, b:-b, b:-b]

            total_psnr += psnr(pred, hr_imgs)
            total_ssim += ssim(pred, hr_imgs)

            # collect samples for W&B logging
            if i < 5:
                samples.append(
                    {
                        "lr": lr_imgs[0].cpu(),
                        "sr": pred[0].cpu(),
                        "hr": hr_imgs[0].cpu(),
                        "fname": fname[0],
                    }
                )

        n = len(benchmark_dl)
        return {
            "psnr": total_psnr / n,
            "ssim": total_ssim / n,
            "samples": samples,
        }

    @torch.no_grad()
    def validate_satellite(self, sat_val_dl, name: str = "DIOR") -> dict:
        from utils.metrics import psnr, ssim, psnr_y, ssim_y
        from data.datasets import generate_lr_on_gpu

        self.model.eval()
        total_psnr_rgb = 0.0
        total_ssim_rgb = 0.0
        total_psnr_y = 0.0
        total_ssim_y = 0.0
        samples = []
        window_size = self.config["window_size"]
        scale = self.config["scale"]

        for i, hr_imgs in enumerate(sat_val_dl):
            hr_imgs = hr_imgs.to(self.device).float()

            # generate LR on GPU — deterministic (no augmentation during val)
            lr_imgs = generate_lr_on_gpu(hr_imgs, scale=scale)

            # pad LR to window size
            _, _, h, w = lr_imgs.shape
            pad_h = (window_size - h % window_size) % window_size
            pad_w = (window_size - w % window_size) % window_size
            if pad_h > 0 or pad_w > 0:
                lr_imgs = F.pad(lr_imgs, (0, pad_w, 0, pad_h), mode="reflect")

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = self.model(lr_imgs).float().clamp(0, 1)

            # crop to HR size
            hr_h, hr_w = hr_imgs.shape[-2], hr_imgs.shape[-1]
            pred = pred[:, :, :hr_h, :hr_w]

            # boundary crop
            b = scale
            pred = pred[:, :, b:-b, b:-b]
            hr_crop = hr_imgs[:, :, b:-b, b:-b]

            total_psnr_rgb += psnr(pred, hr_crop)
            total_ssim_rgb += ssim(pred, hr_crop)
            total_psnr_y += psnr_y(pred, hr_crop)
            total_ssim_y += ssim_y(pred, hr_crop)

            if i < 5:
                samples.append(
                    {
                        "lr": lr_imgs[0, :, :h, :w].cpu(),
                        "sr": pred[0].cpu(),
                        "hr": hr_crop[0].cpu(),
                    }
                )

        n = len(sat_val_dl)
        return {
            "psnr_rgb": total_psnr_rgb / n,
            "ssim_rgb": total_ssim_rgb / n,
            "psnr_y": total_psnr_y / n,
            "ssim_y": total_ssim_y / n,
            "samples": samples,
        }

    # ── main training loop ────────────────
    def fit(self, epochs: int, lr_max: float, lr_min: float, validate_every: int = 1):
        print(f"starting training for {epochs} epochs")
        print(f"SGDR T0={self.config['sgdr_t0']} | LR {lr_max} → {lr_min}")
        print(f"validating every {validate_every} epochs")
        print("-" * 50)

        for epoch in range(self.start_epoch, self.start_epoch + epochs):

            current_lr = self.optimizer.param_groups[0]["lr"]

            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            train_time = time.time() - t0

            # step scheduler after each epoch
            self.scheduler.step()

            log_dict = {
                "train/loss": train_loss,
                "train/lr": current_lr,
                "time/train_epoch": train_time,
                "epoch": epoch,
            }

            if (epoch + 1) % validate_every == 0 or epoch == self.start_epoch:
                t0 = time.time()

                if self.config["mode"] == "satellite":
                    metrics = self.validate_satellite(self.valid_dl, "DIOR")
                    # use Y-channel PSNR as primary metric for satellite
                    primary_psnr = metrics["psnr_y"]
                    val_log = {
                        "val/psnr_rgb": metrics["psnr_rgb"],
                        "val/ssim_rgb": metrics["ssim_rgb"],
                        "val/psnr_y": metrics["psnr_y"],
                        "val/ssim_y": metrics["ssim_y"],
                    }
                else:
                    metrics = self.validate_benchmark(self.valid_dl, "Set5")
                    primary_psnr = metrics["psnr"]
                    val_log = {
                        "val/psnr": metrics["psnr"],
                        "val/ssim": metrics["ssim"],
                    }

                val_time = time.time() - t0
                val_log["time/val_epoch"] = val_time
                val_log["time/total_epoch"] = train_time + val_time
                log_dict.update(val_log)

                if (epoch + 1) % 10 == 0 and "samples" in metrics:
                    self._log_samples(metrics["samples"], epoch)

                if primary_psnr > self.best_psnr:
                    self.best_psnr = primary_psnr
                    self.save_checkpoint(epoch, metrics, tag="best")
                    print(
                        f"epoch {epoch:4d} | loss {train_loss:.4f} | "
                        f"PSNR {primary_psnr:.2f}dB ← best | "
                        f"train {train_time:.0f}s | val {val_time:.0f}s | "
                        f"LR {current_lr:.2e}"
                    )
                else:
                    print(
                        f"epoch {epoch:4d} | loss {train_loss:.4f} | "
                        f"PSNR {primary_psnr:.2f}dB | "
                        f"train {train_time:.0f}s | val {val_time:.0f}s | "
                        f"LR {current_lr:.2e}"
                    )

                self.save_checkpoint(epoch, metrics, tag="latest")
                self.save_checkpoint(epoch, metrics, tag=f"epoch_{epoch:04d}")

            else:
                log_dict["time/total_epoch"] = train_time
                print(
                    f"epoch {epoch:4d} | loss {train_loss:.4f} | "
                    f"train {train_time:.0f}s | LR {current_lr:.2e}"
                )

            wandb.log(log_dict, step=epoch)

        print("-" * 50)
        print(f"training complete. best PSNR: {self.best_psnr:.2f}dB")
