# Medical Image Fusion Project

This project implements and evaluates three state-of-the-art medical image fusion algorithms for combining images from different modalities (CT, MRI, PET, SPECT) to provide more comprehensive diagnostic information.

## Implemented Models

In this project, we've implemented and compared three different medical image fusion methods:

### 1. LRD (Laplacian Re-Decomposition)

LRD is based on multi-scale decomposition using Laplacian pyramids with an additional re-decomposition step to enhance details.

**Key Features:**
- Multi-scale representation using Laplacian pyramid
- Re-decomposition stage to enhance structural details
- Saliency-based fusion rules
- Computationally efficient

### 2. NSST-PAPCNN (Non-Subsampled Shearlet Transform with Parameter-Adaptive Pulse Coupled Neural Network)

This model combines the multi-directional and multi-scale analysis capabilities of NSST with the adaptive fusion capabilities of PAPCNN.

**Key Features:**
- Multi-scale and multi-directional decomposition
- Neural network-based fusion without training requirements
- Parameter-adaptive approach based on spatial frequency
- Good preservation of edge features

### 3. U2Fusion (Unified Unsupervised Image Fusion)

U2Fusion is an advanced model that uses a unified framework for fusion with edge-preserving filtering.

**Key Features:**
- Base and detail layer decomposition using guided filtering
- Saliency-based weight calculation
- Soft-thresholding for detail layer fusion
- Robust performance across various types of medical images

## Dataset

The project uses the Harvard Medical Image Fusion Dataset, which includes:
- CT-MRI pairs
- PET-MRI pairs
- SPECT-MRI pairs

## Evaluation Metrics

The fusion methods are evaluated using:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Computational efficiency

## Project Structure

```
image-fusion-zoom-main/
│
├── fusion_evaluation.py         # Evaluation metrics and utilities
├── fusion_models.py             # Core implementations of fusion models
├── run_lrd_fusion.ipynb         # Notebook for running LRD fusion
├── run_nsst_papcnn_fusion.ipynb # Notebook for running NSST-PAPCNN fusion
├── run_u2fusion_fusion.ipynb    # Notebook for running U2Fusion
├── compare_fusion_models.ipynb  # Notebook for comparing all models
│
├── fused_images/                # Directory for storing fusion results
│   ├── LRD/                     # Results from LRD fusion
│   ├── NSST_PAPCNN/             # Results from NSST-PAPCNN fusion
│   └── U2Fusion/                # Results from U2Fusion
│
└── Medical_Image_Fusion_Methods/ # Source datasets and reference implementations
    └── Havard-Medical-Image-Fusion-Datasets/
        ├── CT-MRI/
        ├── PET-MRI/
        └── SPECT-MRI/
```

## How to Use

1. Set up the virtual environment:
   ```
   conda create -n fusion_env python=3.8
   conda activate fusion_env
   ```

2. Install required packages:
   ```
   pip install numpy matplotlib opencv-python pillow scipy scikit-image tqdm pywavelets notebook pandas seaborn
   ```

3. Run the notebooks:
   - `run_lrd_fusion.ipynb` - Run LRD fusion model
   - `run_nsst_papcnn_fusion.ipynb` - Run NSST-PAPCNN fusion model
   - `run_u2fusion_fusion.ipynb` - Run U2Fusion model
   - `compare_fusion_models.ipynb` - Compare all models

The fusion results will be saved in the `fused_images` directory, organized by model type.
