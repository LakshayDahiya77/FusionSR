# FusionSR: A Hybrid CNN-Transformer for Image Super-Resolution

FusionSR is a high-performance deep learning model for Classical Image Super-Resolution (x4), inspired by the state-of-the-art SwinIR architecture. This project involves designing, implementing, and training a novel hybrid model that leverages the strengths of both Convolutional Neural Networks (CNNs) for efficient local feature extraction and Swin Transformers for global context modeling.

This implementation was developed in PyTorch and trained on an NVIDIA A100 GPU using Google Colab.

##  Architecture

The core of FusionSR is its hybrid design, which intelligently combines different neural network components to maximize performance.

1.  **Shallow Feature Head (CNN):** Instead of a single convolutional layer, FusionSR uses a stack of `ResidualBlocks`. This allows for a more powerful extraction of low-level features like edges and textures before they are passed to the main transformer body.

2.  **Deep Feature Body (Swin Transformer):** The main body consists of 8 architecturally-correct `Swin Transformer Blocks`. This implementation was built from scratch and includes key mechanisms from the original SwinIR paper:
    * **Windowed & Shifted-Window Multi-Head Self-Attention (W-MSA / SW-MSA)** to enable cross-window connections.
    * **Relative Position Bias** to give the model a sense of spatial awareness within each attention window.

3.  **Learnable Feature Fusion:** A dedicated convolutional block is used to intelligently fuse the outputs from the shallow CNN head and the deep Transformer body, allowing the model to learn the optimal combination of local and global features.

4.  **Reconstruction:** The final high-resolution image is reconstructed using an efficient `PixelShuffle` layer.

## Benchmarks

The model was trained on a combined dataset of DIV2K and Flickr2K and evaluated on standard super-resolution benchmarks.

| Model | Dataset | PSNR (dB) | SSIM |
| :--- | :---: | :---: | :---: |
| SwinIR-M (Baseline) | Set14 | 28.08 | 0.7701 |
| **FusionSR (v5)** | **Set14** | **XX.XX** | **.XXXX** |

## Getting Started

### Prerequisites

* Python 3.8+
* PyTorch 2.0+
* CUDA-enabled GPU

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/FusionSR.git](https://github.com/your-username/FusionSR.git)
    cd FusionSR
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    *(You will need to create a `requirements.txt` file containing `torch`, `torchvision`, `einops`, `scikit-image`, `pandas`, `opencv-python`)*

### Training

1.  **Organize Datasets:** Place the DIV2K, Flickr2K, and Set14 datasets in a structured folder.
2.  **Update Paths:** Modify the paths in the configuration cell of the notebook to point to your dataset and project directories.
3.  **Run Notebook:** The project is organized in a Jupyter/Colab notebook. You can execute the cells sequentially to train the model.
    * *(Link to v5 Notebook)*
    * *(Link to v5-finetune Notebook)*

### Inference

An inference cell is provided in the notebook to upscale your own low-resolution images using the trained model.

## Future Work

* **Model Scaling:** Experiment with increasing model capacity (e.g., more blocks, wider dimensions) to further boost performance.
* **EMA Integration:** Resolve library conflicts to re-integrate Exponential Moving Average (EMA) for potentially more stable and higher-quality results.
* **Advanced Loss Functions:** Explore more complex loss functions, such as SSIM/LPIPS loss or frequency-domain loss.

## Acknowledgements

This project is heavily inspired by the original SwinIR paper and its official implementation.
* [SwinIR: Image Restoration Using Swin Transformer (ICCV 2021)](https://arxiv.org/abs/2108.10257)
