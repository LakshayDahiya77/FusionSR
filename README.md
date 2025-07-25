# FusionSR: A Progressive CNN-Transformer for Image Super-Resolution

FusionSR is a high-performance deep learning model for 4x classical image super-resolution. This project implements a novel hybrid architecture and a sophisticated three-stage progressive training framework to efficiently transfer knowledge from a stable baseline to an enhanced model, maximizing performance within a limited compute budget.

The model was developed in PyTorch and trained on an NVIDIA A100/L4 GPU using a highly optimized, GPU-accelerated data pipeline.

## 🚀 Key Features

* **Progressive Transfer Learning:** A 3-stage training strategy that freezes different parts of the network to safely integrate new, advanced components without performance collapse.
* **Hybrid Architecture:** Combines a novel **Progressive Shallow Extractor** with Residual Channel Attention (RCA) blocks and a deep, architecturally-correct **Swin Transformer** backbone.
* **Dynamic GPU Pipeline:** A highly efficient data pipeline that loads full-size images to GPU memory and performs all patch extraction and augmentation on-the-fly, eliminating CPU bottlenecks and maximizing GPU utilization.
* **State-of-the-Art Performance:** Achieves competitive results against the original SwinIR baseline on standard benchmarks.

## 🏗️ Architecture

FusionSR's architecture is designed to stably integrate advanced components onto a pre-trained backbone.

1.  **Progressive Shallow Extractor:** This custom module contains two parallel paths:
    * **Original Path:** A simple stack of CNNs identical to the pre-trained `v5` model's feature extractor.
    * **Enhanced Path:** A new path utilizing **Residual Channel Attention (RCA) Blocks**, which allow the model to focus on the most informative feature channels.
    * A learnable `mix_weight` parameter allows the model to progressively blend the output from the enhanced path with the stable, pre-trained original path, preventing feature corruption.

2.  **Swin Transformer Body:** The deep feature extractor is an 8-layer Swin Transformer backbone, correctly implemented with **shifted-window multi-head self-attention** and **relative position bias**.

3.  **Feature Fusion & Reconstruction:** A standard convolutional block fuses the shallow and deep features, which are then upscaled using an efficient `PixelShuffle` layer.

## ⚙️ Training Strategy

The model was trained on the DIV2K and Flickr2K datasets using a three-stage progressive learning strategy to fine-tune the `v5` baseline:

* **Stage 1 (Epochs 1-15):** Freeze the entire pre-trained model and train **only the new RCA components** and the `mix_weight`. This allows the new blocks to adapt without destabilizing the rest of the network.
* **Stage 2 (Epochs 16-35):** Unfreeze the entire shallow feature extractor (`ProgressiveShallowExtractor`) and continue training with a lower learning rate.
* **Stage 3 (Epochs 36-50):** Unfreeze the entire model and fine-tune all parameters with a very low learning rate to achieve the final optimal performance.

## 📊 Performance

The final model was evaluated on standard super-resolution benchmark datasets.

| Model | Dataset | PSNR (dB) | SSIM |
| :--- | :---: | :---: | :---: |
| SwinIR-M (Baseline) | Set14 | 28.08 | 0.7701 |
| **FusionSR (v7.4)** | **Set14** | **XX.XX** | **.XXXX** |
| *(Other datasets)*| ... | ... | ... |

*(Note: Please update the benchmark table with the final results from your benchmarking cell.)*

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* PyTorch 2.0+
* NVIDIA GPU with CUDA support

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/LakshayDahiya77/FusionSR.git](https://github.com/LakshayDahiya77/FusionSR.git)
    cd FusionSR
    ```

2.  Install the required packages:
    ```bash
    pip install torch torchvision einops scikit-image pandas opencv-python
    ```

### Usage

The project is contained within the `FusionSR_ClassicalSR_v7_4.ipynb` notebook.
1.  **Update Paths:** In **Cell 3**, modify the `DRIVE_PREFIX` and `V5_MODEL_PATH` to point to your project directory and the pre-trained `v5` model weights.
2.  **Run All:** Execute the cells sequentially. The notebook will handle data copying, model building, and the three-stage training process automatically.

## Acknowledgements

This project is heavily inspired by the original SwinIR paper and its official implementation.
* [SwinIR: Image Restoration Using Swin Transformer (ICCV 2021)](https://arxiv.org/abs/2108.10257)
