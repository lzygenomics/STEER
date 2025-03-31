# STEER: a graph attention based Spatial-Temporal Explainable Expert model for RNA velocity inference

STEER is a deep learning framework that leverages spatial-temporal gene expression information and graph attention mechanisms to perform interpretable RNA velocity inference. It supports a variety of modules for training, visualization, prior construction, and utilities tailored to single-cell dynamics.

---

## 🧬 System Requirements (for full functionality)

Some features of STEER require R and the `mclust` package.  
We recommend installing them via conda:

```bash
conda install -c conda-forge r-base=4.3.3 r-mclust=6.1.1
```

---


## 🚀 INSTALL

We **highly recommend** using a virtual environment (e.g., conda) with Python 3.9+.

### 🔧 Step-by-step GPU Setup (CUDA 12.4 + PyTorch 2.4)

```bash

# 0. Some features of STEER require R and the `mclust` package, we recommend installing them via conda:
conda install -c conda-forge r-base=4.3.3 r-mclust=6.1.1

# 1. Install PyTorch with CUDA 12.4 support
pip install torch==2.4.0+cu124 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# 2. Install PyG core libraries compatible with PyTorch 2.4.0 + CUDA 12.4
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install torch-geometric

# 3. Install STEER from GitHub
pip install git+https://github.com/lzygenomics/STEER.git
```

---


## 📦 USAGE

After installation, you can import STEER and use its modules like:

```python
from STEER import model_training_share_neighbor_adata, preprocess, embedding_plot

# Example usage:
adata = preprocess(raw_adata)
model_training_share_neighbor_adata(adata)
embedding_plot(adata)
```

---

## 📁 MODULES

- `training`: Core training routines for RNA velocity prediction
- `prior`: Prior knowledge extraction functions
- `plot`: High-quality plotting utilities for embedding, time, gene expression, etc.
- `utils`: Preprocessing, graph building, simulation, and utility functions

---

## 🧪 REQUIREMENTS

See [`requirements.txt`](./requirements.txt) for full dependencies. Key libraries include:
- `numpy`, `scipy`, `torch`, `scanpy`, `scvelo`, `matplotlib`, `seaborn`

---

## ✍️ CITATION

If you use STEER in your research, please cite:
> _Add your paper link or citation information here_

---

## 👩‍💻 AUTHOR

Developed by [lzygenomics](https://github.com/lzygenomics)

---

## 📫 CONTACT

If you encounter any issues or have questions, feel free to open an issue on GitHub or contact via email: _[lzy_math@163.com]_
