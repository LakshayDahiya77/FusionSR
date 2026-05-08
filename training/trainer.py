import os
import time
import math
import torch
import torch.nn as nn
from utils.metrics import psnr, ssim


# ─────────────────────────────────────────
#  LR Scheduler — cosine decay
# ─────────────────────────────────────────
def cosine_lr(optimizer, epoch, total_epochs, lr_max, lr_min):
    """Cosine annealing without restarts."""
    lr = lr_min + 0.5 * (lr_max - lr_min) * (
        1 + math.cos(math.pi * epoch / total_epochs)
    )
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


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

        os.makedirs(save_dir, exist_ok=True)

    # ── training ──────────────────────────
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for lr_imgs, hr_imgs in self.train_dl:
            lr_imgs = lr_imgs.to(self.device, non_blocking=True)
            hr_imgs = hr_imgs.to(self.device, non_blocking=True)

            # GPU augmentation
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

    # ── validation ────────────────────────
    @torch.no_grad()
    def validate(self, num_samples: int = 5) -> dict:
        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        samples = []

        for i, (lr_imgs, hr_imgs) in enumerate(self.valid_dl):
            lr_imgs = lr_imgs.to(self.device)
            hr_imgs = hr_imgs.to(self.device)

            # pad LR so H and W are divisible by window_size
            _, _, h, w = lr_imgs.shape
            window_size = 8
            pad_h = (window_size - h % window_size) % window_size
            pad_w = (window_size - w % window_size) % window_size
            if pad_h > 0 or pad_w > 0:
                lr_imgs = torch.nn.functional.pad(
                    lr_imgs, (0, pad_w, 0, pad_h), mode="reflect"
                )

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = self.model(lr_imgs).float().clamp(0, 1)

            # crop prediction back to original HR size
            hr_h, hr_w = hr_imgs.shape[-2], hr_imgs.shape[-1]
            pred = pred[:, :, :hr_h, :hr_w]

            hr_imgs = hr_imgs.float()
            total_psnr += psnr(pred, hr_imgs)
            total_ssim += ssim(pred, hr_imgs)

            if i < num_samples:
                samples.append(
                    {
                        "lr": lr_imgs[:, :, :h, :w][0].cpu(),
                        "sr": pred[0].cpu(),
                        "hr": hr_imgs[0].cpu(),
                    }
                )

        n = len(self.valid_dl)
        return {
            "psnr": total_psnr / n,
            "ssim": total_ssim / n,
            "samples": samples,
        }

    # ── checkpoint ────────────────────────
    def save_checkpoint(self, epoch: int, metrics: dict, tag: str = "latest"):
        path = os.path.join(self.save_dir, f"fusionsr_{tag}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "best_psnr": self.best_psnr,
                "config": self.config,
                "metrics": metrics,
            },
            path,
        )

        # upload to W&B as artifact
        artifact = wandb.Artifact(
            name=f"fusionsr-{tag}", type="model", metadata={"epoch": epoch, **metrics}
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    def download_checkpoint(run_path: str, tag: str = "latest") -> str:
        """
        Download checkpoint from W&B.
        run_path: e.g. 'lakshay_dahiya77/FusionSR'
        """
        artifact = wandb.use_artifact(f"fusionsr-{tag}:latest", type="model")
        artifact_dir = artifact.download()
        return os.path.join(artifact_dir, f"fusionsr_{tag}.pt")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.best_psnr = ckpt["best_psnr"]
        self.start_epoch = ckpt["epoch"] + 1
        print(f"resumed from epoch {ckpt['epoch']} | best PSNR {self.best_psnr:.2f}dB")

    # ── W&B image logging ─────────────────
    def _log_samples(self, samples: list, epoch: int):
        panels = []
        for s in samples:
            # upscale LR to HR size for side-by-side comparison
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

            # stack LR (bicubic) | SR | HR side by side
            comparison = torch.cat([lr_up, s["sr"], s["hr"]], dim=2)  # concat on W
            img = comparison.permute(1, 2, 0).numpy()
            panels.append(wandb.Image(img, caption="bicubic | SR | HR"))

        wandb.log({"samples": panels}, step=epoch)

    # ── main loop ─────────────────────────
    def fit(self, epochs: int, lr_max: float, lr_min: float, validate_every: int = 5):

        print(f"starting training for {epochs} epochs")
        print(f"LR: {lr_max} → {lr_min} (cosine)")
        print(f"validating every {validate_every} epochs")
        print("-" * 50)

        for epoch in range(self.start_epoch, self.start_epoch + epochs):

            lr = cosine_lr(
                self.optimizer, epoch, self.start_epoch + epochs, lr_max, lr_min
            )

            # train
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            train_time = time.time() - t0

            log_dict = {
                "train/loss": train_loss,
                "train/lr": lr,
                "time/train_epoch": train_time,
                "epoch": epoch,
            }

            # validate
            if (epoch + 1) % validate_every == 0 or epoch == 0:
                t0 = time.time()
                metrics = self.validate()
                val_time = time.time() - t0

                log_dict.update(
                    {
                        "val/psnr": metrics["psnr"],
                        "val/ssim": metrics["ssim"],
                        "time/val_epoch": val_time,
                        "time/total_epoch": train_time + val_time,
                    }
                )

                if (epoch + 1) % 10 == 0:
                    self._log_samples(metrics["samples"], epoch)

                if metrics["psnr"] > self.best_psnr:
                    self.best_psnr = metrics["psnr"]
                    self.save_checkpoint(epoch, metrics, tag="best")
                    print(
                        f"epoch {epoch:4d} | loss {train_loss:.4f} | "
                        f"PSNR {metrics['psnr']:.2f}dB ← best | "
                        f"SSIM {metrics['ssim']:.4f} | "
                        f"train {train_time:.0f}s | val {val_time:.0f}s"
                    )
                else:
                    print(
                        f"epoch {epoch:4d} | loss {train_loss:.4f} | "
                        f"PSNR {metrics['psnr']:.2f}dB | "
                        f"SSIM {metrics['ssim']:.4f} | "
                        f"train {train_time:.0f}s | val {val_time:.0f}s"
                    )

                self.save_checkpoint(epoch, metrics, tag="latest")

            else:
                log_dict["time/total_epoch"] = train_time
                print(
                    f"epoch {epoch:4d} | loss {train_loss:.4f} | "
                    f"train {train_time:.0f}s | LR {lr:.2e}"
                )

            wandb.log(log_dict, step=epoch)

        print("-" * 50)
        print(f"training complete. best PSNR: {self.best_psnr:.2f}dB")
