import os
import copy
import glob
import torch
import wandb
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import numpy as np
from models.fusionsr import FusionSR, count_parameters
from models.losses import CombinedSRLoss
from training.trainer import Trainer
from data.datasets import make_train_dl, make_benchmark_dl, generate_lr_on_gpu


# ─────────────────────────────────────────────────────────────────────────────
#  UNIFIED DATASET STORAGE PIPELINE (RAM-SAFE)
# ─────────────────────────────────────────────────────────────────────────────

class UnifiedHRDataset(Dataset):
    """Memory-safe dataset class for streaming massive HR-only folders with precise count filtering."""
    def __init__(self, root_dir, patch_size=128, scale=4, filters=None):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.scale = scale
        self.patch_hr = patch_size * scale
        
        raw_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            raw_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
            raw_paths.extend(glob.glob(os.path.join(root_dir, '**', ext.upper()), recursive=True))
        
        # Apply Dictionary-based Filtering with Limits
        if filters and isinstance(filters, dict):
            filtered_paths = []
            for filter_str, max_limit in filters.items():
                f_lower = filter_str.lower()
                # Find all matching files and sort them deterministically
                matches = sorted([p for p in raw_paths if f_lower in os.path.basename(p).lower()])
                
                # Apply the specific cutoff limit if one is provided
                if max_limit is not None:
                    matches = matches[:max_limit]
                    
                filtered_paths.extend(matches)
            
            # Remove potential duplicates while preserving order
            self.img_paths = list(dict.fromkeys(filtered_paths))
        else:
            self.img_paths = sorted(list(set(raw_paths)))
            
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images found matching filters {filters} in {root_dir}")
        else:
            print(f"UnifiedHRDataset initialized with {len(self.img_paths)} total images.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                w, h = img.size
                
                if w < self.patch_hr or h < self.patch_hr:
                    img = img.resize((max(w, self.patch_hr), max(h, self.patch_hr)), Image.BICUBIC)
                    w, h = img.size

                x0 = random.randint(0, w - self.patch_hr)
                y0 = random.randint(0, h - self.patch_hr)
                cropped_img = img.crop((x0, y0, x0 + self.patch_hr, y0 + self.patch_hr))
                
                if random.random() > 0.5:
                    cropped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT)
                rot = random.choice([0, 90, 180, 270])
                if rot != 0:
                    cropped_img = cropped_img.rotate(rot)
                
                hr_tensor = torch.from_numpy(np.array(cropped_img, dtype=np.uint8, copy=True))
                hr_tensor = hr_tensor.permute(2, 0, 1).float() / 255.0
                return hr_tensor
        except Exception as e:
            return self.__getitem__(random.randint(0, len(self.img_paths) - 1))


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "channels": 180,
    "num_groups": 6,
    "num_rcab": 6,
    "window_size": 16,
    "num_heads": 6,
    "scale": 4,
    "ffn_expansion": 2.0,
    "oca_overlap": 4,
    "total_epochs": 150,
    "start_epoch": 0,
    "lr_max": 3e-4,
    "batch_size": 16,
    "patch_lr": 128,
    "num_workers": 4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "warmup_epochs": 8,
    "min_lr": 1e-7,
    "allow_tf32": True,
    "matmul_precision": "high",
    "use_compile": False,
    "compile_mode": "max-autotune",
    "compile_fullgraph": False,
    "compile_dynamic": False,
    
    # EMA settings
    "use_ema": False,
    "ema_decay": 0.999,
    
    # Dataset routing toggles
    "use_unified_dataset": False,
    "unified_hr_dir": "",
    
    # Traditional split fallbacks
    "train_hr_dirs": [],
    "train_lr_dirs": [],
    "val_hr_dir": "",
    "val_lr_dir": "",
    
    "wandb_entity": "lakshay_dahiya77",
    "wandb_project": "FusionSR-v4",
    "wandb_run": "v4-phase1",
    "wandb_run_id": None,
    "resume": None,
    "save_dir": "/content/checkpoints",
}


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    config = copy.deepcopy(CONFIG)

    if config.get("allow_tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    matmul_precision = config.get("matmul_precision")
    if matmul_precision:
        torch.set_float32_matmul_precision(matmul_precision)

    print(f"\ndevice: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name} ({props.total_memory / 1024**3:.1f}GB)")

    wandb.login()
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

    assert config["val_hr_dir"], "Validation paths are mandatory."
    assert config["val_lr_dir"], "Validation paths are mandatory."

    # Routing Dataloaders via flag
    if config.get("use_unified_dataset", False):
        print(f"Initializing Unified Storage-backed stream from: {config['unified_hr_dir']}")
        train_ds = UnifiedHRDataset(
            root_dir=config["unified_hr_dir"],
            patch_size=config["patch_lr"],
            scale=config["scale"],
            filters=config.get("unified_filters", None) 
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=True,
            drop_last=True
        )
    else:
        assert config["train_hr_dirs"], "Legacy mode requires valid train_hr_dirs"
        train_dl = make_train_dl(
            hr_dirs=config["train_hr_dirs"],
            lr_dirs=config["train_lr_dirs"] or None,
            patch_lr=config["patch_lr"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            scale=config["scale"],
        )

    valid_dl = make_benchmark_dl(
        hr_dir=config["val_hr_dir"],
        lr_dir=config["val_lr_dir"],
    )

    print(f"train batches: {len(train_dl)} | valid images: {len(valid_dl)}")

    model = FusionSR(
        channels=config["channels"],
        num_groups=config["num_groups"],
        num_rcab=config["num_rcab"],
        window_size=config["window_size"],
        num_heads=config["num_heads"],
        scale=config["scale"],
        ffn_expansion=config["ffn_expansion"],
        oca_overlap=config["oca_overlap"],
    ).to(device)

    if config.get("use_compile", False) and hasattr(torch, "compile"):
        model = torch.compile(
            model,
            mode=config.get("compile_mode", "max-autotune"),
            fullgraph=config.get("compile_fullgraph", False),
            dynamic=config.get("compile_dynamic", False),
        )

    print(f"parameters: {count_parameters(model) / 1e6:.2f}M")

    ema_model = None
    if config.get("use_ema", False):
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(config["ema_decay"]))

    loss_fn = CombinedSRLoss(pixel_weight=1.0, use_perceptual=False).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr_max"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999),
    )

    # Intercepting training iterations for HR-only batches to step generation on GPU
    if config.get("use_unified_dataset", False):
        class UnifiedStepTrainer(Trainer):
            # FIXED: Removed 'epoch' from the arguments
            def train_epoch(self): 
                # Custom processing loop wrapper that generates LR targets on GPU dynamically
                self.model.train()
                epoch_loss = 0.0
                for batch_idx, hr_tensors in enumerate(self.train_dl):
                    hr_tensors = hr_tensors.to(self.device, non_blocking=True)
                    lr_tensors = generate_lr_on_gpu(hr_tensors, scale=self.config["scale"])
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    with torch.autocast("cuda", dtype=torch.bfloat16): # Added autocast for speed
                        pred = self.model(lr_tensors)
                        loss, loss_dict = self.loss_fn(pred, hr_tensors)
                        
                    loss.backward()
                    
                    if self.config["grad_clip"] > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"])
                    
                    self.optimizer.step()
                    
                    if hasattr(self, "ema_model") and self.ema_model is not None:
                        self.ema_model.update_parameters(self.model)
                        
                    epoch_loss += loss.item()
                    
                return epoch_loss / len(self.train_dl)
        
        trainer = UnifiedStepTrainer(
            model=model, loss_fn=loss_fn, optimizer=optimizer,
            train_dl=train_dl, valid_dl=valid_dl, config=config,
            device=device, save_dir=config["save_dir"]
        )

    if ema_model is not None:
        trainer.ema_model = ema_model

    if config["resume"]:
        resume_path = config["resume"]
        if "/" in resume_path:
            print(f"downloading artifact: {resume_path}")
            artifact = wandb.use_artifact(resume_path, type="model")
            artifact_dir = artifact.download()
            pt_files = glob.glob(os.path.join(artifact_dir, "*.pt"))
            assert pt_files
            ckpt_path = pt_files[0]
        else:
            ckpt_path = resume_path
        trainer.load_checkpoint(ckpt_path)

    trainer.fit()

    print("\n" + "=" * 60)
    print("post-training benchmark evaluation")
    print("=" * 60)
    m = trainer.validate_benchmark(valid_dl, "Urban100")
    print(f"  Urban100 — PSNR(Y): {m['psnr']:.2f}dB | SSIM(Y): {m['ssim']:.4f}")

    wandb.finish()
    print("\ndone.")

if __name__ == "__main__":
    main()
