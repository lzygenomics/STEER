<div class="hero" markdown>

<p class="hero-kicker">Environment Setup</p>

# Installation

<p class="hero-subtitle">
Install STEER in a clean environment with Python 3.9 or newer. Some features also depend on R and
<code>mclust</code>. The examples below show tested configurations rather than strict version locks.
</p>

<div class="hero-actions" markdown>

[Open Quick Start](quickstart.md)
[View Source](https://github.com/lzygenomics/STEER)

</div>

</div>

<div class="grid cards" markdown>

-   :material-check-decagram-outline: **Core requirements**

    ---

    Python 3.9+, PyTorch, PyG, `torch-scatter`, and `torch-sparse`.

-   :material-language-r: **Optional R support**

    ---

    Some STEER features additionally require `r-base` and `r-mclust`.

-   :material-alert-circle-outline: **Compatibility rule**

    ---

    Match your CUDA, PyTorch, and PyG wheels carefully within the same environment.

</div>

## Recommended environment

### Example GPU setup

Tested during development with a configuration such as:

- **GPU**: NVIDIA 3090
- **CUDA**: 12.4
- **PyTorch**: 2.4.0

```bash
# Create and activate environment
conda create -n steer python=3.10
conda activate steer

# Optional R dependencies
conda install -c conda-forge r-base=4.3.3 r-mclust=6.1.1

# Install PyTorch with CUDA 12.4 support
pip install torch==2.4.0+cu124 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install PyG dependencies
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install torch-geometric

# Install STEER
pip install git+https://github.com/lzygenomics/STEER.git
```

### Tested newer setup

STEER has also been tested in a newer environment such as:

- **GPU**: NVIDIA L20
- **CUDA**: 12.8
- **PyTorch**: 2.8.0
- **Python**: 3.10

```bash
conda create -n steer python=3.10
conda activate steer

conda install -c conda-forge r-base=4.3.3 r-mclust=6.1.1

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install torch-geometric
```

## Notes

The environment versions listed above reflect configurations used during development and testing. They should be regarded as **recommended examples rather than strict version requirements**.

In general, STEER should work as long as the following packages are installed in a mutually compatible way:

- `torch`
- `torch-geometric`
- `torch-scatter`
- `torch-sparse`

## Additional dependencies

For the broader Python environment, please also refer to the project requirements file. Key libraries include:

- `numpy`
- `scipy`
- `scanpy`
- `scvelo`
- `matplotlib`
- `seaborn`

See the repository `requirements.txt` for the full dependency list.

## Troubleshooting

### R dependency issues

If some STEER features fail because of missing R dependencies, make sure the following are available in your environment:

- `r-base`
- `r-mclust`

### PyG installation issues

If `torch-scatter` or `torch-sparse` fail to install, check that:

1. your CUDA version matches the installed PyTorch wheel
2. the PyG wheel URL matches your PyTorch version
3. you are installing into the intended virtual environment

### Verify installation

After installation, open Python and test whether the main dependencies can be imported successfully.
