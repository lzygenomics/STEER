# STEER: A graph attention based Spatial-Temporal Explainable Expert model for RNA velocity inference

STEER is a deep learning framework that leverages spatial-temporal gene expression information and graph attention mechanisms to perform interpretable RNA velocity inference. It supports a variety of modules for training, visualization, prior construction, and utilities tailored to single-cell and spatial dynamics.

---

## 🚀 INSTALL

We **highly recommend** using a virtual environment (e.g., conda) with Python 3.9+.

### 🔧 Step-by-step GPU Setup (e.g., 3090 GPU + CUDA 12.4 + PyTorch 2.4)

```bash
# 0. Some features of STEER require R and the `mclust` package, we recommend installing them via conda:
conda install -c conda-forge r-base=4.3.3 r-mclust=6.1.1

# 1. Install PyTorch with CUDA 12.4 support
pip install torch==2.4.0+cu124 torchvision==0.19.0 torchaudio==2.4.0 --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)

# 2. Install PyG core libraries compatible with PyTorch 2.4.0 + CUDA 12.4
pip install torch-scatter -f [https://data.pyg.org/whl/torch-2.4.0+cu124.html](https://data.pyg.org/whl/torch-2.4.0+cu124.html)
pip install torch-sparse -f [https://data.pyg.org/whl/torch-2.4.0+cu124.html](https://data.pyg.org/whl/torch-2.4.0+cu124.html)
pip install torch-geometric

# 3. Install STEER from GitHub
pip install git+[https://github.com/lzygenomics/STEER.git](https://github.com/lzygenomics/STEER.git)

```

*Note: This step describes the approach used to configure the environment at the time of development. The specified package versions are not strict requirements. As long as the key packages are installed in a compatible manner, newer versions may also be used.*

### Test on Newer Versions (e.g., L20 GPU + CUDA 12.8 + PyTorch 2.8 + Python 3.10)

```bash
conda create -n steer python=3.10
conda activate steer
conda install -c conda-forge r-base=4.3.3 r-mclust=6.1.1

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)
pip install torch-scatter -f [https://data.pyg.org/whl/torch-2.8.0+cu128.html](https://data.pyg.org/whl/torch-2.8.0+cu128.html)
pip install torch-sparse -f [https://data.pyg.org/whl/torch-2.8.0+cu128.html](https://data.pyg.org/whl/torch-2.8.0+cu128.html)
pip install torch-geometric

```

---

## 📖 Tutorials & Example Usage

To help you seamlessly integrate STEER into your research, we provide comprehensive Jupyter Notebooks and pipelines covering from raw data preprocessing to downstream velocity inference.

### 1. Core Model Pipeline

We provide an end-to-end example notebook that walks you through the entire STEER workflow:

* 🚀 **[STEER Core Pipeline Demo](./tutorials/demo_tutorial.ipynb)**: Step-by-step interactive usage of the STEER framework.

### 2. Generating Spliced/Unspliced Matrices (Data Preprocessing)

A practical challenge in spatial RNA velocity is obtaining spliced and unspliced count matrices from raw sequencing data. We provide dedicated guidelines and pipelines for major spatial platforms:

* 🧬 **[Slide-seq Pipeline](./tutorials/raw_data_processing/Slide-seq/)**: An end-to-end workflow resolving 15bp-to-14bp barcode mismatches and running `velocyto`.
* 🔬 **[10x Visium Pipeline](./tutorials/raw_data_processing/Visium/)**: Standard workflows bridging `velocyto` with Seurat outputs.
* 💻 **[Stereo-seq Pipeline](./tutorials/raw_data_processing/Stereo-seq/)**: Integration workflows based on the official `stereopy` ecosystem.

---

## 🧪 REQUIREMENTS

See [`requirements.txt`](./requirements.txt) for full dependencies. Key libraries include:

* `numpy`, `scipy`, `torch`, `scanpy`, `scvelo`, `matplotlib`, `seaborn`

---

## ✍️ CITATION

If you use STEER in your research, please cite our work:

> *[ ]*

---

## 👩‍💻 AUTHOR

Developed by [lzygenomics](https://github.com/lzygenomics)

---

## 📫 CONTACT

If you encounter any issues or have questions, feel free to open an issue on GitHub or contact via email: **lzy_math@163.com**

### Happy Researching! Wishing you health and success in your projects. ✨