"""
inference.py — Upscale a single image using FusionSR.

Usage:
    python inference.py --input image.jpg --output upscaled.png --model classical
    python inference.py --input image.jpg --checkpoint /path/to/checkpoint.pt
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from huggingface_hub import hf_hub_download
from models.fusionsr import FusionSR

V2_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "channels": 96,
    "num_groups": 6,
    "num_rcab": 6,
    "window_size": 8,
    "num_heads": 4,
    "scale": 4,
    "ffn_expansion": 2.0,
}

V4_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "channels": 180,
    "num_groups": 6,
    "num_rcab": 6,
    "window_size": 16,
    "num_heads": 6,
    "scale": 4,
    "ffn_expansion": 2.0,
    "oca_overlap": 4,
}

V5_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "channels": 162,
    "num_groups": 8,
    "num_heads": 6,
    "scale": 4,
    "ffn_expansion": 2.0,
    "window_size": 8,  # used for external padding calculation
}

def load_model(
    model_identifier: str,
    device: torch.device,
    model_config: dict,
) -> torch.nn.Module:
    """Loads from HF Hub if given 'classical'/'satellite', else treats as local path."""
    if model_identifier in ["classical", "satellite"]:
        filename = f"fusionsr-v2-{model_identifier}.pt"
        ckpt_path = hf_hub_download(
            repo_id="lakshaydahiya/FusionSR-v2",
            filename=filename,
        )
        print(f"loaded {filename} from HuggingFace Hub")
    else:
        ckpt_path = model_identifier
        print(f"loaded local checkpoint from {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = FusionSR(**model_config).to(device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()
    return model


def download_artifact(artifact_name: str) -> str:
    import wandb

    api = wandb.Api()
    artifact = api.artifact(artifact_name, type="model")
    ckpt_dir = artifact.download()
    pt_files = list(Path(ckpt_dir).glob("*.pt"))
    assert len(pt_files) > 0
    return str(pt_files[0])


@torch.no_grad()
def upscale(model, img_tensor, device, scale=4):
    """Run SR model on image tensor. Window padding handled by model."""
    lr = img_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        sr = model(lr).float().clamp(0, 1)

    return sr.squeeze(0)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if args.checkpoint:
        ckpt_path = args.checkpoint
    elif args.model:
        ckpt_path = args.model
    else:
        raise ValueError("Specify --checkpoint or --model")

    if args.version == "v4" and args.model in ["classical", "satellite"]:
        raise ValueError("HF 'classical'/'satellite' checkpoints are v2; use --version v2 or a v4 checkpoint")

    if args.version == "v2" or (args.version is None and args.model in ["classical", "satellite"]):
        model_config = V2_CONFIG
    elif args.version == "v4":
        model_config = V4_CONFIG
    else:
        # Default to v5 for local checkpoints
        model_config = V5_CONFIG

    # load model
    print(f"loading checkpoint: {ckpt_path}")
    model = load_model(ckpt_path, device, model_config)
    print(f"model loaded — scale: {model_config['scale']}×")

    # load input image
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {args.input}")

    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    print(f"input: {input_path.name} ({w}×{h})")

    # convert to tensor
    img_tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(
        2, 0, 1
    )  # [3, H, W]

    # upscale
    print("upscaling...")
    sr_tensor = upscale(
        model,
        img_tensor,
        device,
        scale=model_config["scale"],
    )

    # convert back to PIL and save
    sr_np = (sr_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    sr_img = Image.fromarray(sr_np)

    # resolve output path
    if args.output:
        out_path = args.output
    else:
        out_path = str(input_path.parent / f"{input_path.stem}_4x{input_path.suffix}")

    sr_img.save(out_path)
    ow, oh = sr_img.size
    print(f"saved: {out_path} ({ow}×{oh})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FusionSR inference — 4× image upscaling"
    )
    parser.add_argument("--input", type=str, required=True, help="input image path")
    parser.add_argument("--output", type=str, default=None, help="output image path")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["classical", "satellite"],
        help="which model to use (downloads from W&B)",
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=["v2", "v4", "v5"],
        default=None,
        help="model version to use (defaults: v2 for HF models, v5 otherwise)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="local checkpoint path (overrides --model)",
    )
    args = parser.parse_args()
    main(args)
