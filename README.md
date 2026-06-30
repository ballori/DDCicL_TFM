# DCDicL_TFM: Gravitational Wave Reconstruction
This repository contains the source code for my Master's Thesis (TFM) in Advanced Physics (Astrophysics) at the Universitat de València. The project focuses on utilizing Deep Convolutional Dictionary Learning (DCDicL) and Sparse Dictionary Learning techniques to isolate and reconstruct gravitational wave signals, with a particular focus on core-collapse supernovae.

## Repository Structure
* `dataset_generation.py`: Scripts for processing the raw wave catalogs and generating the training datasets.
* `train_single.py` & `train_for_study.py`: Training routines for the 1D convolutional architectures. Includes scripts for single model training and extensive parametric studies to optimize hyperparameters (filters, kernels, learning rates).
* `inference_single.py` & `inference_for_study.py`: Signal processing and reconstruction scripts to evaluate the model's performance on test data.
* `study.py`: Orchestrates the automated execution of the parametric studies.
* `ccphen.py` & `libccphen.so`: Core physical phenomenology utilities and compiled libraries for signal manipulation.
* `config.yaml`: Centralized configuration for hyperparameters, model architecture, and path management.

## Important Technical Notes
* **Numerical Precision:** The training scripts are strictly configured to run in pure `float32` precision. Automatic Mixed Precision (AMP) logic has been intentionally excluded to maintain absolute numerical stability during the dictionary learning processes and gradient scaling.
* **Dataset Handling:** The main gravitational wave catalog (`data/SNWaveCatalog/GWDB.h5`) and the PyTorch model weights (`.pt`) are tracked locally and excluded from this repository. Ensure you have the local database correctly placed in the `data/` directory before initiating dataset generation.

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/ballori/DDCicL_TFM.git](https://github.com/ballori/DDCicL_TFM.git)
   cd DDCicL_TFM
