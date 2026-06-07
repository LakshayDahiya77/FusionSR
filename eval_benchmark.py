import os
import torch
import wandb
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from models.fusionsr import FusionSR
from utils.metrics import psnr_y, ssim_y

# Default V5 Configuration
V5_CONFIG = {
    "channels": 162,
    "num_groups": 8,
    "num_heads": 6,
    "scale": 4,
    "ffn_expansion": 2.0,
    "use_hfeb": True,
    "use_hybrid_ca": True,
    "use_mswa": True,
    "use_tdca": True,
    "tdca_num_tokens": 64,
    "tdca_interval": 2,
    "hf_scale_init": 0.01,
}

class BenchmarkDataset(Dataset):
    """Handles standard and 'x4' suffix variations in test sets."""
    def __init__(self, hr_dir, lr_dir):
        self.hr_files = sorted(Path(hr_dir).glob("*.png"))
        self.lr_dir = Path(lr_dir)
        assert len(self.hr_files) > 0, f"No HR images found in {hr_dir}"

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        lr_path = self.lr_dir / hr_path.name
        
        if not lr_path.exists():
            lr_path = self.lr_dir / f"{hr_path.stem}x4{hr_path.suffix}"

        hr = np.array(Image.open(hr_path).convert("RGB"), dtype=np.uint8)
        lr = np.array(Image.open(lr_path).convert("RGB"), dtype=np.uint8)

        hr = torch.from_numpy(hr).permute(2, 0, 1).float() / 255.0
        lr = torch.from_numpy(lr).permute(2, 0, 1).float() / 255.0

        return lr, hr, hr_path.name

def run_benchmark(config):
    """
    config should contain:
    - wandb_entity
    - wandb_project
    - run_name
    - artifact_name
    - datasets: dict of {name: {"hr": path, "lr": path}}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    run = wandb.init(
        entity=config["wandb_entity"], 
        project=config["wandb_project"], 
        name=config["run_name"], 
        job_type="benchmark"
    )
    
    print(f"Downloading artifact: {config['artifact_name']}")
    artifact = run.use_artifact(config["artifact_name"], type="model")
    artifact_dir = artifact.download()
    ckpt_paths = list(Path(artifact_dir).glob("*.pt"))
    assert len(ckpt_paths) > 0, "No checkpoint files found in artifact!"
    
    print("Loading model...")
    model = FusionSR(**V5_CONFIG).to(device)
    ckpt = torch.load(str(ckpt_paths[0]), map_location=device, weights_only=False)
    
    if "model" in ckpt:
        # Strip DataParallel "module." prefix if it exists in the checkpoint
        state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(ckpt)
        
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    scale = V5_CONFIG["scale"]
    
    for ds_name, paths in config["datasets"].items():
        print(f"\nEvaluating {ds_name}...")
        dataset = BenchmarkDataset(paths["hr"], paths["lr"])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
        
        total_psnr = 0.0
        total_ssim = 0.0
        samples_to_log = []

        with torch.no_grad():
            for i, (lr, hr, fname) in enumerate(dataloader):
                lr = lr.to(device)
                hr = hr.to(device)
                
                # FusionSR handles padding internally via _pad_to_window
                with torch.autocast("cuda", dtype=torch.float16):
                    pred = model(lr).float().clamp(0, 1)
                
                # Calculate exact theoretical output size based on LR input
                expected_h, expected_w = lr.shape[-2] * scale, lr.shape[-1] * scale
                
                # Crop HR to discard the non-divisible modulo pixels (e.g. 481 -> 480)
                hr_cropped_to_scale = hr[:, :, :expected_h, :expected_w]
                
                # Standard SR boundary crop (remove edges)
                pred_crop = pred[:, :, scale:-scale, scale:-scale]
                hr_crop = hr_cropped_to_scale[:, :, scale:-scale, scale:-scale]
                
                psnr_val = psnr_y(pred_crop, hr_crop)
                ssim_val = ssim_y(pred_crop, hr_crop)
                
                total_psnr += psnr_val
                total_ssim += ssim_val

                # Grab first 3 images for W&B media tab
                if i < 3:
                    lr_up = torch.nn.functional.interpolate(lr, size=(expected_h, expected_w), mode="bicubic", align_corners=False).clamp(0, 1)
                    comparison = torch.cat([lr_up[0].cpu(), pred[0].cpu(), hr_cropped_to_scale[0].cpu()], dim=2)
                    img_np = (comparison.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    samples_to_log.append(wandb.Image(img_np, caption=f"{fname[0]} - Bicubic | SR | HR"))

        avg_psnr = total_psnr / len(dataloader)
        avg_ssim = total_ssim / len(dataloader)
        
        print(f"{ds_name} -> PSNR(Y): {avg_psnr:.2f}dB | SSIM(Y): {avg_ssim:.4f}")
        
        wandb.log({
            f"{ds_name}/PSNR_Y": avg_psnr,
            f"{ds_name}/SSIM_Y": avg_ssim,
            f"{ds_name}/Samples": samples_to_log
        })
        
    wandb.finish()
