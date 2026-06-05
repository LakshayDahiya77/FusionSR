import os
import time
import math
import torch
import torch.nn as nn
import wandb

from utils.metrics import psnr_y, ssim_y
from data.datasets import gpu_augment, generate_lr_on_gpu


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
        save_dir: str = "/content/checkpoints",
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.config = config
        self.device = device
        self.save_dir = save_dir

        self.grad_clip = config.get("grad_clip", 1.0)

        gpu_cap = torch.cuda.get_device_capability(device)
        if gpu_cap[0] >= 8:
            self.amp_dtype = torch.bfloat16
            self.scaler = torch.amp.GradScaler("cuda", enabled=False)
            print(f"AMP: bfloat16 (GPU capability {gpu_cap[0]}.{gpu_cap[1]})")
        else:
            self.amp_dtype = torch.float16
            self.scaler = torch.amp.GradScaler("cuda")
            print(f"AMP: float16 (GPU capability {gpu_cap[0]}.{gpu_cap[1]})")

        self.best_psnr = 0.0
        self.start_epoch = config.get("start_epoch", 0)

        self.scheduler = None

        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in self.train_dl:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                lr_imgs, hr_imgs = batch
                lr_imgs = lr_imgs.to(self.device, non_blocking=True)
                hr_imgs = hr_imgs.to(self.device, non_blocking=True)
            else:
                hr_imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
                hr_imgs = hr_imgs.to(self.device, non_blocking=True)
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

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(self.train_dl)

    @torch.no_grad()
    def validate_benchmark(self, benchmark_dl, name: str) -> dict:
        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        samples = []
        scale = self.config["scale"]

        for i, (lr_imgs, hr_imgs, fname) in enumerate(benchmark_dl):
            lr_imgs = lr_imgs.to(self.device)
            hr_imgs = hr_imgs.to(self.device).float()

            with torch.autocast("cuda", dtype=self.amp_dtype):
                pred = self.model(lr_imgs).float().clamp(0, 1)

            hr_h, hr_w = hr_imgs.shape[-2], hr_imgs.shape[-1]
            pred = pred[:, :, :hr_h, :hr_w]

            b = scale
            pred_crop = pred[:, :, b:-b, b:-b]
            hr_crop = hr_imgs[:, :, b:-b, b:-b]

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

    def _log_samples(self, samples: list):
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

    def save_checkpoint(self, epoch: int, metrics: dict, tag: str = "latest"):
        path = os.path.join(self.save_dir, f"fusionsr_{tag}.pt")

        model_state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()

        ckpt = {
            "epoch": epoch,
            "model": model_state,
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
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt["model"])

        self.start_epoch = self.config["start_epoch"]
        self.best_psnr = 0.0

        m = ckpt.get("metrics", {})
        print(f"── checkpoint loaded ──")
        print(f"  source epoch: {ckpt.get('epoch', '?')}")
        print(f"  PSNR (Y): {m.get('psnr', 'N/A')}")
        print(f"  SSIM (Y): {m.get('ssim', 'N/A')}")
        print(f"── training config ──")
        print(f"  start_epoch: {self.start_epoch}")
        print(f"  lr_max: {self.config['lr_max']}")

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
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return cosine * (1.0 - min_factor) + min_factor

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambda,
            last_epoch=self.start_epoch * steps_per_epoch - 1,
        )

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

            t0 = time.time()
            train_loss = self.train_epoch()
            train_time = time.time() - t0

            t0 = time.time()
            metrics = self.validate_benchmark(self.valid_dl, "Set5")
            val_time = time.time() - t0

            val_psnr = metrics["psnr"]
            val_ssim = metrics["ssim"]

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

            if metrics.get("samples"):
                self._log_samples(metrics["samples"])

            is_best = val_psnr > self.best_psnr
            if is_best:
                self.best_psnr = val_psnr
                self.save_checkpoint(epoch, metrics, tag="best")

            self.save_checkpoint(epoch, metrics, tag="latest")

            best_marker = " ← best" if is_best else ""
            print(
                f"epoch {epoch:4d} | loss {train_loss:.4f} | "
                f"PSNR(Y) {val_psnr:.2f}dB | SSIM(Y) {val_ssim:.4f} | "
                f"train {train_time:.0f}s | val {val_time:.0f}s | "
                f"LR {current_lr:.2e}{best_marker}"
            )

        print("-" * 60)
        print(f"training complete. best PSNR(Y): {self.best_psnr:.2f}dB")