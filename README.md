# STEER: a graph attention based Spatial-Temporal Explainable Expert model for RNA velocity inference

STEER is a deep learning framework that leverages spatial-temporal gene expression information and graph attention mechanisms to perform interpretable RNA velocity inference. It supports a variety of modules for training, visualization, prior construction, and utilities tailored to single-cell dynamics.


## 🚀 INSTALL

We **highly recommend** using a virtual environment (e.g., conda) with Python 3.9+.

### 🔧 Step-by-step GPU Setup (3090 GPU + CUDA 12.4 + PyTorch 2.4)

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

This step describes the approach the author used to configure the environment at the time. It is provided as a reference, and the specified package versions are not strict requirements. As long as the key packages are installed in a compatible manner, newer versions may also be used.
```

### New Version CUDA test (L20 GPU + CUDA 12.8 + PyTorch 2.8 + Python 3.10)
```bash

conda create -n steer python=3.10
conda install -c conda-forge r-base=4.3.3 r-mclust=6.1.1
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install torch-geometric

```


---


## 📦 Example Usage

After installation, you can import STEER and use its modules like:

```python
import scanpy as sc
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

#----- Infer Initial Cell Group & Gene level expression trendency as prior -----#
keep_ngene = 1000
# Add random cluster to compare
# adata = assign_random_clusters_to_cells(adata, labels=['label1', 'label2', 'label3', 'label4', 'label5'])
df = steer.add_annotations_to_df(df, result_adata, annotation_cols = ['pred_cluster', 'celltype'] )
pretrain_df = steer.use_scaled_orig(df, use_orig = False)

# Calculate Entropy for each labels
Entropy_df = steer.grid_us_space_extended(pretrain_df, bins=5, label_cols = ['pred_cluster', 'celltype'], min_ncell = 5)
result_adata = steer.add_entropy_to_adata(result_adata, Entropy_df, top_n = result_adata.n_vars)
# optional, clean_df do not need in model but may need in visualize
clean_df = pretrain_df
result_adata = steer.pred_regulation_anndata(result_adata)
# save results
# remove the gene which no prior for all cell
result_adata.var.loc[np.all(result_adata.layers['pred_cell_type'] == "Auto", axis=0), 'is_velocity_gene'] = False
result_adata.write(RESULT_PATH + 'prior_adata.h5ad')

#----- Fine-Tuning Phase for dynamic learning -----#
prior_adata = result_adata.copy()
raw_adata = sc.read_h5ad(DATA_PATH)
# del raw_adata.obsm['X_pca']
assert all(prior_adata.obs_names == raw_adata.obs_names), "Observation names are not aligned!"
assert all(prior_adata.var_names == raw_adata.var_names), "Variable names are not aligned!"
raw_adata.layers['pred_cell_type'] = prior_adata.layers['pred_cell_type']
raw_adata.obsm['X_pre_embed'] = prior_adata.obsm['X_pre_embed']
raw_adata.obs['pred_cluster'] = prior_adata.obs['pred_cluster'].astype(int)
velo_adata = raw_adata[:,prior_adata.var['is_velocity_gene']].copy() # type: ignore
del prior_adata, raw_adata
df, adjacency_matrix, velo_adata = steer.preprocess_anndata(velo_adata, 
                                                      npc=30, 
                                                      NUM_AD_NEIGH = 30, 
                                                      SMOOTH_NEIGH = s,
                                                      use_us=False)
# Load data
dataset = steer.preload_datasets_all_genes_anndata(df = df, MODEL_MODE = 'whole', adata = velo_adata)
# Construct PyG Data object
pyg_data = steer.create_pyg_data(dataset, adjacency_matrix, normalize = True)
# Train model —— to see the embedding of selected genes
velo_adata = steer.model_training_share_neighbor_adata(device = device, 
                                                device2 = device2,
                                                pyg_data = pyg_data, 
                                                MODEL_MODE = 'whole', 
                                                adata = velo_adata, 
                                                NUM_LOSS_NEIGH = 30, 
                                                max_n_cluster = num_cluster,
                                                path = RESULT_PATH)
result_adata = velo_adata.copy()
# If need to normalize each pred_vu and pred_vs to l2 norm be 1
result_adata = steer.normalize_l2_anndata(result_adata)
result_adata = steer.clean_anndata(result_adata)
result_adata.write(RESULT_PATH + 'final_adata.h5ad')

#----- More details and visualizations will be updated later -----#
```

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
