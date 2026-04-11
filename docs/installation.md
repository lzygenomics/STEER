# Installation

STEER currently recommends using a virtual environment with **Python 3.9+**.[^install]

## Recommended setup

### 1. Create an environment

```bash
conda create -n steer python=3.10
conda activate steer
```

### 2. Install optional R dependencies

Some STEER features require **R** and the `mclust` package.[^install]

```bash
conda install -c conda-forge r-base=4.3.3 r-mclust=6.1.1
```

### 3. Install PyTorch

The README gives an example for **CUDA 12.4 + PyTorch 2.4.0**:

```bash
pip install torch==2.4.0+cu124 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
```

It also shows a newer tested path using **CUDA 12.8 + PyTorch 2.8.0**.[^install]

### 4. Install PyG dependencies

For the CUDA 12.4 example:

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install torch-geometric
```

### 5. Install STEER

```bash
pip install git+https://github.com/lzygenomics/STEER.git
```

## Notes

!!! note
    The repository explicitly says the listed package versions are **not strict requirements**. They reflect the development environment and can be replaced by newer compatible versions.[^install]

!!! warning
    PyTorch and PyG compatibility is the part most likely to fail first. If installation breaks, align **Python**, **CUDA**, **PyTorch**, and **PyG wheel versions** before debugging STEER itself.

## Verify the installation

Try importing the package in Python:

```python
import steer
print("STEER imported successfully")
```

If that works, continue to the [Quick Start](quickstart.md).

[^install]: STEER's README includes Python, R, PyTorch, PyG, and installation guidance, plus a newer tested environment example. citeturn259328view0turn259328view3
