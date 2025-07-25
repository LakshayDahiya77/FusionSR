# FusionSR: A Progressive CNN-Transformer for Image Super-Resolution

FusionSR is a high-performance deep learning model for 4x classical image super-resolution. This project implements a novel hybrid architecture and a sophisticated three-stage progressive training framework to efficiently transfer knowledge from a stable baseline to an enhanced model, maximizing performance within a limited compute budget.

The model was developed in PyTorch and trained on an NVIDIA A100/L4 GPU using a highly optimized, GPU-accelerated data pipeline.

## 🚀 Key Features

* **Progressive Transfer Learning:** A 3-stage training strategy that freezes different parts of the network to safely integrate new, advanced components without performance collapse.
* **Hybrid Architecture:** Combines a novel **Progressive Shallow Extractor** with Residual Channel Attention (RCA) blocks and a deep, architecturally-correct **Swin Transformer** backbone.
* **Dynamic GPU Pipeline:** A highly efficient data pipeline that loads full-sized images to the GPU and performs all patch extraction and augmentations on-the-fly, eliminating CPU bottlenecks.
* **Comprehensive Benchmarking:** The final model was successfully benchmarked across multiple standard datasets.

## 📊 Performance

The final model was evaluated on standard super-resolution benchmark datasets. While it establishes a strong performance baseline for a custom architecture, it does not surpass the highly-optimized SwinIR model.

| Model               | Dataset   | PSNR (dB) |  SSIM  |
| :------------------ | :-------: | :-------: | :----: |
| SwinIR-M (Baseline) |   Set14   |   28.08   | 0.7701 |
| **FusionSR (Final)**| **Set5** | **25.150**| **0.7466** |
| **FusionSR (Final)**| **Set14** | **23.533**| **0.6849** |
| **FusionSR (Final)**| **BSD100**| **24.544**| **0.6830** |
| **FusionSR (Final)**|**Urban100**| **21.591**| **0.6484** |

## 🏗️ Architecture

FusionSR's architecture is designed to stably integrate advanced components onto a pre-trained backbone.

1.  **Progressive Shallow Extractor:** This custom module contains two parallel paths: an original path preserved from a pre-trained baseline and an enhanced path utilizing **Residual Channel Attention (RCA) Blocks**. A learnable `mix_weight` allows the model to progressively blend the output from the two paths.

2.  **Swin Transformer Body:** The deep feature extractor is an 8-layer Swin Transformer backbone, correctly implemented with **shifted-window multi-head self-attention** and **relative position bias**.

3.  **Feature Fusion & Reconstruction:** A standard convolutional block fuses the shallow and deep features, which are then upscaled using an efficient `PixelShuffle` layer.

## ⚙️ Training Strategy

The model was trained on the DIV2K and Flickr2K datasets using a three-stage progressive learning strategy to fine-tune a stable `v5` baseline:

* **Stage 1:** Freeze the pre-trained model and train **only the new RCA components**.
* **Stage 2:** Unfreeze and train the **entire shallow feature extractor**.
* **Stage 3:** Unfreeze and fine-tune the **entire model** with a low learning rate.
