# MaxSR: Image Super-Resolution Using Improved MaxViT

This repository implements **MaxSR**, a **state-of-the-art image super-resolution model** based on the paper [MaxSR: Image Super-Resolution Using Improved MaxViT](https://arxiv.org/pdf/2307.07240). The model leverages **MaxViT-based multi-axis self-attention** to enhance image resolution efficiently.

🚀 **This project is designed to work on limited-resource environments!**  
I have carefully optimized and restructured the implementation to make it accessible for users with constrained hardware.

---

## 📌 Features

✅ **Multi-Axis Transformer (MaxViT-based) for Super-Resolution**  
✅ **Three Model Variants**: Standard, Tiny, and Super-Tiny  
✅ **Optimized for Limited VRAM** (Runs on low-end GPUs and CPUs)  
✅ **Generic Modular Implementation** (Components structured in `src/components/`)  
✅ **Evaluation Metrics**: PSNR & SSIM  
✅ **Jupyter Notebook for Results Visualization**  

---

## 🏗️ Model Variants

| Model Name        | Size     | Performance | Resource Requirement |
|------------------|---------|------------|----------------------|
| **MaxSR**        | Large   | Best       | High VRAM & Compute  |
| **MaxSR-Tiny**   | Medium  | Balanced   | Lower VRAM Needed    |
| **MaxSR-SuperTiny** | Small  | Lightweight | Extremely Low VRAM   |

🔹 **MaxSR-SuperTiny** is my custom variant, designed to run with minimal GPU/memory usage while still maintaining quality.

---

## 📂 Project Structure

```plaintext
MaxSR
│── src/
|   │── config/            # Configuration files for all 3 models
│   ├── components/         # Generic reusable modules
│   ├── models/             # Model implementations
│   ├── evaluation/         # PSNR, SSIM calculation
│   ├── utils/              # Preprocessing & utilities
│   ├── notebooks/          # Jupyter Notebooks for visualization
│   ├── preprocessing/      # image extraction and patches
│── main.py                 # main script
│── README.md               # This file
```

---

## 🛠️ Installation

**please ensure you have div2k dataset on your machine**
```bash
git clone https://github.com/RyanWri/MaxSR.git
cd MaxSR
pip install -r requirements.txt
python src/main.py 
```

## 📊 Results Presentation (Jupyter Notebook)
The presentation Jupyter notebook (src/notebooks/presentation.ipynb) demonstrates the performance of all three models (MaxSR, MaxSR-Tiny, MaxSR-SuperTiny) on multiple images.

To run and visualize:
```
jupyter notebook src/notebooks/presentation.ipynb
```

Inside the notebook:
1. Images are loaded from the dataset.
2. Each model generates super-resolved outputs.
3. PSNR/SSIM metrics are computed for quality assessment.

## 🛠️ Configuration Files
Each model variant has its own configuration file:

```plaintext
config/
│── maxsr.yaml         # Config for the full MaxSR model
│── maxsr_tiny.yaml    # Config for the Tiny variant
│── maxsr_super_tiny.yaml # Config for the Super-Tiny variant
```

These configs define:
1. Model architecture details (e.g., number of blocks, layers, attention heads)
2. Training hyperparameters (e.g., batch size, learning rate)
3. Inference settings (e.g., output resolution, normalization)

To use a custom configuration:
```
python train.py --config config/maxsr_tiny.yaml
```

## 🔬 Model Evaluation (PSNR & SSIM)
To measure the quality of super-resolution output, we use PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index Measure):

1. **PSNR**: Measures the pixel-wise similarity between the super-resolved and ground truth images. Higher PSNR means better quality.
2. **SSIM**: Evaluates the structural and perceptual similarity between two images, capturing human visual perception better than PSNR.

## 🙌 Acknowledgments
***This project was heavily optimized to work in resource-constrained environments.
A significant effort was made to restructure the implementation, reduce memory usage, and enable it to run on low-end GPUs.***

For any questions, feel free to open an issue or reach out.

### ⭐ If you find this project useful, please consider giving it a star!
