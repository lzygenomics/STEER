# STEER: a graph attention based Spatial-Temporal Explainable Expert model for RNA velocity inference

STEER is a deep learning framework that leverages spatial-temporal gene expression information and graph attention mechanisms to perform interpretable RNA velocity inference. It supports a variety of modules for training, visualization, prior construction, and utilities tailored to single-cell dynamics.


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


## 📦 Example Usage

After installation, you can import STEER and use its modules like:

```python
import steer

# Input and Clean data, adata is the prepared anndata format with unsplice and splice
df, adjacency_matrix, adata = steer.preprocess_anndata(adata,npc=30,NUM_AD_NEIGH = 30)
# Prepare data for model
dataset = steer.preload_datasets_all_genes_anndata(df = df, MODEL_MODE = 'pretrain', adata = adata)
# Construct PyG Data object
pyg_data = steer.create_pyg_data(dataset, adjacency_matrix, normalize = True)

#----- Pre-Train Phase for Celluar context learning -----#
result_adata = steer.model_training_share_neighbor_adata(device = 'cuda:0', 
                                                device2 = 'cuda:1',
                                                pyg_data = pyg_data, 
                                                MODEL_MODE = 'pretrain', 
                                                adata = adata, 
                                                NUM_LOSS_NEIGH = 30, 
                                                max_n_cluster = num_cluster,
                                                path = RESULT_PATH)
# Cluster cells based on Pre-Train embedding
result_adata = steer.mclust_R(result_adata, num_cluster=num_cluster)
# Plot predicted clusters and previous cell type annotation
sc.pl.embedding(result_adata, basis = 'umap', color=['pred_cluster', 'celltype'], show=False, save = '_NumClus_'+str(num_cluster)+'.svg')
sc.pl.embedding(result_adata, basis = 'X_umap_pre_embed', color=['pred_cluster','celltype'], show=False, save = '_NumClus_'+str(num_cluster)+'.svg')
# Save Pre-Train results and release GPU memory
result_adata.write(RESULT_PATH+ 'Num_'+str(num_cluster)+'_pretrain_adata.h5ad')
df.to_csv(RESULT_PATH+'Num_' +str(num_cluster) + '_pretrain_df.csv')
torch.cuda.empty_cache()
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
