import scvelo as scv
import pandas as pd
import anndata
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch.utils.data import Dataset
from scipy.integrate import solve_ivp
import rpy2
from sklearn.decomposition import PCA
import scanpy as sc
import anndata as ad
from scipy.sparse import coo_matrix
from scvelo.tools.utils import cosine_correlation
from scvelo.core import get_n_jobs, parallelize
from torch_geometric.utils import dense_to_sparse, to_torch_coo_tensor,to_dense_adj, get_laplacian
from torch_geometric.data import Data
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr
#sys.path.append('/HOME/scz3472/run/GATVelo/GATVelo_project/')
from ..plot import velocity_graph
import matplotlib.pyplot as plt
import seaborn as sns
import torch.backends.cudnn as cudnn
# cudnn.deterministic = True
# cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def df_to_adata(df, concatenate=False):
    # Pivot the DataFrame to create matrices for spliced and unspliced counts
    spliced = df.pivot(index='cellID', columns='gene_name', values='splice').fillna(0)
    unspliced = df.pivot(index='cellID', columns='gene_name', values='unsplice').fillna(0)
    # Ensure spliced and unspliced matrices have the same genes (columns) and cells (rows)
    spliced, unspliced = spliced.align(unspliced, join='inner', axis=0)
    # Create an AnnData object with the spliced counts
    adata = anndata.AnnData(X=spliced.values)
    adata.layers['spliced'] = spliced.values
    adata.layers['unspliced'] = unspliced.values
    adata.obs_names = spliced.index
    adata.var_names = spliced.columns
    if concatenate:
        # Concatenate spliced and unspliced matrices
        concat_data = pd.concat([unspliced, spliced], axis=1)
        # Perform PCA on the concatenated data
        pca = PCA(n_components=50)
        pca_result = pca.fit_transform(concat_data)
        # Add PCA results to the AnnData object
        adata.obsm['X_pca'] = pca_result
    return adata

# def add_moments_to_df(df, adata):
#     # Extract Ms and Mu from the AnnData object
#     Ms = pd.DataFrame(adata.layers['Ms'], index=adata.obs_names, columns=adata.var_names)
#     Mu = pd.DataFrame(adata.layers['Mu'], index=adata.obs_names, columns=adata.var_names)
    
#     # Add Ms and Mu to the original DataFrame
#     df['Ms'] = df.apply(lambda row: Ms.at[row['cellID'], row['gene_name']], axis=1)
#     df['Mu'] = df.apply(lambda row: Mu.at[row['cellID'], row['gene_name']], axis=1)
    
#     return df
def add_moments_to_df(df, adata):
    # Extract Ms and Mu from the AnnData object
    Ms = pd.DataFrame(adata.layers['Ms'], index=adata.obs_names, columns=adata.var_names)
    Mu = pd.DataFrame(adata.layers['Mu'], index=adata.obs_names, columns=adata.var_names)
    # Reset index with a specific name
    Ms.reset_index(inplace=True)
    Mu.reset_index(inplace=True)
    # Rename the index column
    Ms.rename(columns={'index': 'cellID'}, inplace=True)
    Mu.rename(columns={'index': 'cellID'}, inplace=True)
    # Melt Ms and Mu to long format for efficient merging
    Ms_melted = Ms.melt(id_vars='cellID', var_name='gene_name', value_name='Ms')
    Mu_melted = Mu.melt(id_vars='cellID', var_name='gene_name', value_name='Mu')
    # Merge Ms and Mu with the original DataFrame
    df = df.merge(Ms_melted, on=['cellID', 'gene_name'], how='left')
    df = df.merge(Mu_melted, on=['cellID', 'gene_name'], how='left')
    return df

def update_moments_df(df, adata):

    # Drop the old Mu and Ms
    df = df.drop(columns=['Mu', 'Ms'])
    # Extract Ms and Mu from the AnnData object
    Ms = pd.DataFrame(adata.layers['Ms'], index=adata.obs_names, columns=adata.var_names)
    Mu = pd.DataFrame(adata.layers['Mu'], index=adata.obs_names, columns=adata.var_names)
    
    # Reset index with a specific name
    Ms.reset_index(inplace=True)
    Mu.reset_index(inplace=True)
    
    # Rename the index column
    Ms.rename(columns={'index': 'cellID'}, inplace=True)
    Mu.rename(columns={'index': 'cellID'}, inplace=True)
    
    # Melt Ms and Mu to long format for efficient merging
    Ms_melted = Ms.melt(id_vars='cellID', var_name='gene_name', value_name='Ms')
    Mu_melted = Mu.melt(id_vars='cellID', var_name='gene_name', value_name='Mu')
    
    # Merge Ms and Mu with the original DataFrame
    df = df.merge(Ms_melted, on=['cellID', 'gene_name'], how='left')
    df = df.merge(Mu_melted, on=['cellID', 'gene_name'], how='left')
    
    df['unsplice'] = df['Mu']
    df['splice'] = df['Ms']

    return df, adata

def convert_anndata(data):

    fix_keep = ['cellID', 'gene_name', 'unsplice', 'splice', 'orig_unsplice', 'orig_splice']

    # Convert matrices to DataFrames
    unspliced_matrix = pd.DataFrame(data.layers['Mu'], index=data.obs.index, columns=data.var.index)
    spliced_matrix = pd.DataFrame(data.layers['Ms'], index=data.obs.index, columns=data.var.index)
    orig_unspliced_matrix = pd.DataFrame(data.layers['unspliced'].toarray() if issparse(data.layers['unspliced']) else data.layers['unspliced'], 
                                         index=data.obs.index, columns=data.var.index)
    orig_spliced_matrix = pd.DataFrame(data.layers['spliced'].toarray() if issparse(data.layers['spliced']) else data.layers['spliced'], 
                                       index=data.obs.index, columns=data.var.index)

    # Melt the unspliced and spliced matrices to long format
    id_column = 'cellID' if 'cellID' in unspliced_matrix.index.names else 'index'
    unspliced_matrix = unspliced_matrix.reset_index()
    spliced_matrix = spliced_matrix.reset_index()
    orig_unspliced_matrix = orig_unspliced_matrix.reset_index()
    orig_spliced_matrix = orig_spliced_matrix.reset_index()
    
    unspliced_long = unspliced_matrix.melt(id_vars=id_column, var_name='gene_name', value_name='unsplice')
    spliced_long = spliced_matrix.melt(id_vars=id_column, var_name='gene_name', value_name='splice')
    orig_unspliced_long = orig_unspliced_matrix.melt(id_vars=id_column, var_name='gene_name', value_name='orig_unsplice')
    orig_spliced_long = orig_spliced_matrix.melt(id_vars=id_column, var_name='gene_name', value_name='orig_splice')    

    # Merge the long data matrices on 'id_column' and 'gene_name'
    combined_long = pd.merge(unspliced_long, spliced_long, on=[id_column, 'gene_name'], how='left')
    orig_combined_long = pd.merge(orig_unspliced_long, orig_spliced_long, on=[id_column, 'gene_name'], how='left')
    combined_long = pd.merge(combined_long, orig_combined_long, on=[id_column, 'gene_name'], how='left')

    # Extract the obs metadata
    obs_metadata = data.obs.copy()

    # Check if 'cellID' exists in obs_metadata, otherwise use the index
    if 'cellID' not in obs_metadata.columns:
        if 'cellID' not in obs_metadata.index.name:
            obs_metadata['cellID'] = obs_metadata.index

    # Merge the combined long data matrix with the obs metadata
    combined_df = pd.merge(combined_long, obs_metadata, left_on=id_column, right_on='cellID', how='left')

    # Add the var metadata to the combined DataFrame
    var_metadata = data.var.reset_index().rename(columns={'index': 'gene_name'})
    combined_df = pd.merge(combined_df, var_metadata, on='gene_name', how='left')

    if 'index' in combined_df.columns:
        # Drop the unnecessary 'index' column after merging
        combined_df = combined_df.drop(columns=['index'])

    # Filter the combined DataFrame by fix_keep list
    combined_df = combined_df[fix_keep]

    # Refresh the index
    combined_df.reset_index(drop=True, inplace=True)

    # Add 'Mu' and 'Ms' columns
    combined_df['Mu'] = combined_df['unsplice']
    combined_df['Ms'] = combined_df['splice']

    return combined_df

def adjancency_and_sequence(df, genes, NUM_AD_NEIGH):
    subset_df = df[df['gene_name'].isin(genes)]
    
    # Create a pivot table for both splice and unsplice values
    splice_df = subset_df.pivot_table(index='cellID', columns='gene_name', values='splice', aggfunc='first').fillna(0)
    unsplice_df = subset_df.pivot_table(index='cellID', columns='gene_name', values='unsplice', aggfunc='first').fillna(0)
    
    # Concatenate the splice and unsplice values to have a 2N dimension vector for each cell
    combined_df = pd.concat([splice_df, unsplice_df], axis=1)
    
    # Apply KNN to find the nearest neighbors for each cell
    knn = NearestNeighbors(n_neighbors=NUM_AD_NEIGH, metric='cosine')
    knn.fit(combined_df)
    distances, indices = knn.kneighbors(combined_df)
    
    # Retrieve the cellID index from the combined_df DataFrame
    cell_ids = combined_df.index.tolist()
    
    # Create the adjacency matrix as a NumPy array
    adjacency_matrix = np.zeros((len(cell_ids), len(cell_ids)), dtype=np.int32)
    
    # Fill the adjacency matrix with 1 for the nearest neighbors
    for i, neighbors in enumerate(indices):
        adjacency_matrix[i, neighbors] = 1
    np.fill_diagonal(adjacency_matrix, 1)
    
    # Map cell names to IDs and vice versa
    cell_name_to_id = {name: idx for idx, name in enumerate(cell_ids)}
    
    return adjacency_matrix, cell_name_to_id

def preprocess(df, genes, Process = True, SMOOTH = True, NUM_AD_NEIGH = 30, SMOOTH_NEIGH = 100, use_us = False, adj_con = True, adj = 'connectivities'):
    if Process: 
        # if has been deal, reconver the original
        df['unsplice'] = df['orig_unsplice']
        df['splice'] = df['orig_splice']
        # Drop the old Mu and Ms
        df = df.drop(columns=['Mu', 'Ms'])
    else:
        # if first deal, keep the original
        df['orig_unsplice'] = df['unsplice']
        df['orig_splice'] = df['splice']
    # cellID must be a integers!
        # Check if any 'cellID' starts with 'cell_'
    if isinstance(df['cellID'].iloc[0], str):
        # Remove 'cell_' prefix from each entry in 'cellID' column
        df['cellID'] = df['cellID'].str.replace('cell_', '')
        # Convert the 'cellID' column to integers
        df['cellID'] = df['cellID'].astype('int64')
    else:
        print("No 'cellID' values start with 'cell_', so no changes are made.")

    # Remove genes
    df = df[df['gene_name'].isin(genes)].reset_index(drop=True)

    adata = df_to_adata(df, concatenate=use_us)
    # sc.pp.neighbors(adata, n_pcs=30, n_neighbors=SMOOTH_NEIGH)
    # Preprocess and compute moments using scvelo
    scv.pp.moments(adata, n_pcs=30, n_neighbors=SMOOTH_NEIGH)

    # Smooth the u and s
    if SMOOTH:
        # Add the computed moments to the original DataFrame
        df = add_moments_to_df(df, adata)
        # Perform smooth
        df['unsplice'] = df['Mu']
        df['splice'] = df['Ms']
    
    if adj_con:
        adata2 = df_to_adata(df, concatenate=use_us)
        sc.pp.neighbors(adata2, n_pcs=30, n_neighbors=NUM_AD_NEIGH, metric='cosine')
        # Convert non-zero elements to 1
        binary_connectivities = adata2.obsp[adj].copy()
        binary_connectivities.data = np.ones_like(binary_connectivities.data)

        # Ensure the diagonal elements are set to 1
        n_cells = binary_connectivities.shape[0]
        binary_connectivities.setdiag(1)

        # Convert to integer type
        binary_connectivities = binary_connectivities.astype(int)

        # Optionally convert to dense format to inspect
        adjacency_matrix = binary_connectivities.toarray()
    else:
        # Get the adjancey matrix
        adjacency_matrix, cell_name_to_id = adjancency_and_sequence(df, genes, NUM_AD_NEIGH = NUM_AD_NEIGH)
    
    # Normalize u and s
    df['unsplice'] =  df.groupby('gene_name')['unsplice'].transform(lambda x: x / x.max())
    df['splice'] =  df.groupby('gene_name')['splice'].transform(lambda x: x / x.max())
    
    return df, adjacency_matrix

def us_moments(adata, npc = 50, n_neighbor = 100, neighbor_metric = 'cosine'):
    # Extract 'spliced' and 'unspliced' layers
    spliced = adata.layers['spliced'].toarray() if hasattr(adata.layers['spliced'], 'toarray') else adata.layers['spliced']
    unspliced = adata.layers['unspliced'].toarray() if hasattr(adata.layers['unspliced'], 'toarray') else adata.layers['unspliced']

    # Combine 'spliced' and 'unspliced' layers by concatenating along the feature axis (genes)
    combined = np.concatenate([spliced, unspliced], axis=1)

    # Create a new AnnData object with the combined data
    combined_adata = sc.AnnData(X=combined)

    # Perform PCA on the combined data
    sc.tl.pca(combined_adata, n_comps = npc)

    # Store the PCA results back in the original AnnData object
    adata.obsm['X_pca_combined'] = combined_adata.obsm['X_pca']
    sc.pp.neighbors(adata, n_pcs=npc, n_neighbors=n_neighbor, use_rep='X_pca_combined', metric=neighbor_metric)
    # sc.pp.neighbors(adata, n_pcs=npc, n_neighbors=n_neighbor, use_rep='X_pca_combined')
    # Calculate moments using the combined PCA representation
    scv.pp.moments(adata, n_pcs=npc, n_neighbors=n_neighbor, use_rep='X_pca_combined')
    return adata

def construct_momented_adj(adata, npc = 50, NUM_AD_NEIGH = 30, neighbor_metric = 'cosine', graph_adj = 'connectivities', use_us = True):
    if use_us:
        # Concatenate Mu and Ms layers
        Mu = adata.layers['Mu']
        Ms = adata.layers['Ms']
        concatenated_data = np.concatenate((Mu, Ms), axis=1)
        
        # Create a new AnnData object with the combined data
        combined_adata = sc.AnnData(X=concatenated_data)

        # Perform PCA on the combined data
        sc.tl.pca(combined_adata, n_comps = npc)
        adata.obsm['X_pca_moments'] = combined_adata.obsm['X_pca']
        sc.pp.neighbors(adata, n_pcs=npc, n_neighbors=NUM_AD_NEIGH, use_rep='X_pca_moments', metric=neighbor_metric)
    else:
        sc.tl.pca(adata, n_comps = npc, layer = 'Ms')
        sc.pp.neighbors(adata, n_pcs=npc, n_neighbors=NUM_AD_NEIGH, use_rep='X_pca', metric=neighbor_metric)
    # Convert non-zero elements to 1
    binary_connectivities = adata.obsp[graph_adj].copy()
    binary_connectivities.data = np.ones_like(binary_connectivities.data)

    # Ensure the diagonal elements are set to 1
    binary_connectivities.setdiag(1)

    # Convert to integer type
    binary_connectivities = binary_connectivities.astype(int)

    # Optionally convert to dense format to inspect
    adjacency_matrix = binary_connectivities.toarray()

    return adata, adjacency_matrix


def construct_latent_adj(adata, npc = 50, NUM_AD_NEIGH = 30, graph_adj = 'connectivities'):

    sc.pp.neighbors(adata, n_neighbors=NUM_AD_NEIGH, use_rep='X_pre_embed')
    # Convert non-zero elements to 1
    binary_connectivities = adata.obsp[graph_adj].copy()
    binary_connectivities.data = np.ones_like(binary_connectivities.data)

    # Ensure the diagonal elements are set to 1
    binary_connectivities.setdiag(1)

    # Convert to integer type
    binary_connectivities = binary_connectivities.astype(int)

    # Optionally convert to dense format to inspect
    adjacency_matrix = binary_connectivities.toarray()

    return adata, adjacency_matrix

def preprocess_anndata_with_custom_data(adata, 
                                layer_mapping=None, 
                                pca_key='X_pca',
                                NUM_AD_NEIGH=30, 
                                neighbor_metric='euclidean', 
                                graph_adj='connectivities'):
    """
    通用预处理接口：使用预先计算好的 PCA 和 图层数据 (Mu, Ms, spliced, unspliced)
    来构建 STEER 模型所需的输入。
    
    参数:
    ----------
    adata : AnnData
        包含预计算数据对象的 AnnData。
    layer_mapping : dict, optional
        定义外部数据层到 STEER 内部名称的映射。
        默认映射适配刚才保存的 H5AD 结构：
        - 'Ms': 模型用的平滑 Spliced -> 'spliced_imputed'
        - 'Mu': 模型用的平滑 Unspliced -> 'unspliced_imputed'
        - 'spliced': 原始 Spliced -> 'spliced_original'
        - 'unspliced': 原始 Unspliced -> 'unspliced_original'
    pca_key : str
        使用的 PCA 坐标键名，默认为 'X_pca'。
    """
    
    # 默认映射配置 (适配之前生成的 H5AD)
    if layer_mapping is None:
        layer_mapping = {
            'Ms': 'spliced_imputed',
            'Mu': 'unspliced_imputed',
            'spliced': 'spliced_original',
            'unspliced': 'unspliced_original'
        }

    adata.obs.index.name = 'cellID'
    adata.var.index.name = 'gene_name'

    print("--- Preprocessing with custom provided layers ---")
    
    # 1. 映射数据层 (Layer Mapping)
    # 将外部定义的层名映射到 convert_anndata 所需的标准名称 (Ms, Mu, spliced, unspliced)
    for internal_name, source_name in layer_mapping.items():
        if source_name in adata.layers:
            print(f"  Mapping '{source_name}' -> '{internal_name}'")
            adata.layers[internal_name] = adata.layers[source_name].copy()
        elif source_name == 'X': # 允许映射到 X
            print(f"  Mapping 'X' -> '{internal_name}'")
            adata.layers[internal_name] = adata.X.copy()
        else:
            # 如果是原始计数缺失，报错；如果是平滑数据缺失，尝试回退
            if internal_name in ['spliced', 'unspliced']:
                 # 尝试找不带 _original 后缀的标准名
                 fallback = internal_name
                 if fallback in adata.layers:
                     print(f"  '{source_name}' not found, using standard '{fallback}'")
                     adata.layers[internal_name] = adata.layers[fallback]
                 else:
                     raise ValueError(f"Critical: Source layer '{source_name}' for '{internal_name}' not found.")
            else:
                print(f"  Warning: Source layer '{source_name}' for '{internal_name}' not found. Falling back to X.")
                adata.layers[internal_name] = adata.X.copy()

    # 2. 构建/复用邻接矩阵 (Neighbor Graph)
    # 直接使用预先计算好的 Embedding (PCA)
    if pca_key not in adata.obsm:
        raise ValueError(f"Missing '{pca_key}' in adata.obsm. Cannot compute neighbors.")
    
    print(f"  Computing neighbors using provided embedding '{pca_key}' (k={NUM_AD_NEIGH})...")
    sc.pp.neighbors(adata, n_neighbors=NUM_AD_NEIGH, use_rep=pca_key, metric=neighbor_metric, key_added='adj')

    # 3. 提取邻接矩阵 (Adjacency Matrix) 为 Dense 格式
    binary_connectivities = adata.obsp['adj_' + graph_adj].copy()
    binary_connectivities.data = np.ones_like(binary_connectivities.data)
    binary_connectivities.setdiag(1)
    binary_connectivities = binary_connectivities.astype(int)
    adjacency_matrix = binary_connectivities.toarray()

    # 4. 生成 DataFrame (调用现有的 convert_anndata)
    # 此时 adata 已经具备了 convert_anndata 所需的所有标准层名
    print("  Converting AnnData to DataFrame...")
    df = convert_anndata(adata) 

    # 5. 归一化 (最大值归一化)
    # 对平滑后的数据列进行 [0,1] 缩放，适合深度学习输入
    print("  Normalizing Ms/Mu columns to [0,1]...")
    df['unsplice'] = df.groupby('gene_name')['unsplice'].transform(lambda x: x / x.max() if x.max() != 0 else x)
    df['splice'] = df.groupby('gene_name')['splice'].transform(lambda x: x / x.max() if x.max() != 0 else x)
    
    return df, adjacency_matrix, adata

def preprocess_anndata(adata, npc = 30, NUM_AD_NEIGH = 30, SMOOTH_NEIGH = 100, use_us = True, neighbor_metric='cosine', graph_adj ='connectivities',latent_graph = False, moments_adj = True, smooth=True):
    adata.obs.index.name = 'cellID'
    adata.var.index.name = 'gene_name'
    if smooth:
        # Smooth Phase
        if use_us:
            adata = us_moments(adata=adata, npc=npc, n_neighbor=SMOOTH_NEIGH, neighbor_metric = neighbor_metric)
        else:
            sc.pp.pca(adata, n_comps=npc, use_highly_variable=False)
            sc.pp.neighbors(adata, n_pcs=npc, n_neighbors=SMOOTH_NEIGH, metric = neighbor_metric)
            scv.pp.moments(adata, n_pcs = npc, n_neighbors = SMOOTH_NEIGH)

    # Convert Anndata to Dataframe Phase
    df = convert_anndata(adata)
    
    if latent_graph:
        adata, adjacency_matrix =  construct_latent_adj(adata, npc = npc, NUM_AD_NEIGH = NUM_AD_NEIGH, graph_adj = graph_adj)
    else:
        if moments_adj:
            # Construct Moments-based adjancency matrix
            adata, adjacency_matrix =  construct_momented_adj(adata, npc = npc, NUM_AD_NEIGH = NUM_AD_NEIGH, neighbor_metric = neighbor_metric, graph_adj = graph_adj, use_us = use_us)
        else:
            if use_us:
                sc.pp.neighbors(adata, n_pcs=npc, n_neighbors=NUM_AD_NEIGH, use_rep='X_pca_combined', metric = neighbor_metric,key_added='adj')
            else:
                sc.pp.neighbors(adata, n_pcs=npc, n_neighbors=NUM_AD_NEIGH, use_rep='X_pca', metric = neighbor_metric,key_added='adj')
            # Convert non-zero elements to 1
            binary_connectivities = adata.obsp['adj_'+graph_adj].copy()
            binary_connectivities.data = np.ones_like(binary_connectivities.data)
            # Ensure the diagonal elements are set to 1
            binary_connectivities.setdiag(1)
            # Convert to integer type
            binary_connectivities = binary_connectivities.astype(int)
            # Optionally convert to dense format to inspect
            adjacency_matrix = binary_connectivities.toarray()
    # Normalize u and s
    df['unsplice'] = df.groupby('gene_name')['unsplice'].transform(lambda x: x / x.max() if x.max() != 0 else x)
    df['splice'] = df.groupby('gene_name')['splice'].transform(lambda x: x / x.max() if x.max() != 0 else x)
    
    return df, adjacency_matrix, adata

def preprocess_anndata_spatial(adata, npc=30, NUM_AD_NEIGH=30, SMOOTH_NEIGH=100, use_us=True, neighbor_metric='cosine', 
                       graph_adj='connectivities', latent_graph=False, moments_adj=True, smooth=True, 
                       use_spatial=True, spatial_neighbors=8, spatial_key='spatial', combine_mode='union',spatial_first = True):
    """
    Preprocess the AnnData object and compute adjacency matrices based on latent or moment-based graphs.
    Optionally include spatial adjacency and combine it with the original adjacency using union or overlap.

    Parameters:
        adata: AnnData object
        npc: int, number of principal components
        NUM_AD_NEIGH: int, number of neighbors for adjacency calculation
        SMOOTH_NEIGH: int, number of neighbors for smoothing
        use_us: bool, whether to use unspliced data
        neighbor_metric: str, metric for neighbors calculation
        graph_adj: str, type of adjacency ('connectivities' or others)
        latent_graph: bool, whether to use latent graph
        moments_adj: bool, whether to use moments-based adjacency
        smooth: bool, whether to smooth the data
        use_spatial: bool, whether to calculate spatial adjacency
        spatial_neighbors: int, number of KNN neighbors for spatial adjacency calculation
        spatial_key: str, key in adata.obsm where spatial coordinates are stored
        combine_mode: str, 'union' or 'overlap' to combine spatial and original adjacency

    Returns:
        df: DataFrame, processed AnnData as a DataFrame
        adjacency_matrix: np.ndarray, combined adjacency matrix
        adata: AnnData object
    """
    adata.obs.index.name = 'cellID'
    adata.var.index.name = 'gene_name'
    
    # Smooth Phase
    if smooth:
        if use_us:
            adata = us_moments(adata=adata, npc=npc, n_neighbor=SMOOTH_NEIGH, neighbor_metric=neighbor_metric)
        else:
            sc.pp.pca(adata, n_comps=npc, use_highly_variable=False)
            sc.pp.neighbors(adata, n_pcs=npc, n_neighbors=SMOOTH_NEIGH, metric=neighbor_metric)
            scv.pp.moments(adata, n_pcs=npc, n_neighbors=SMOOTH_NEIGH)

    # Convert AnnData to DataFrame Phase
    df = convert_anndata(adata)
    
    # Construct original adjacency matrix
    if latent_graph:
        adata, original_adjacency = construct_latent_adj(adata, npc=npc, NUM_AD_NEIGH=NUM_AD_NEIGH, graph_adj=graph_adj)
    elif moments_adj:
        adata, original_adjacency = construct_momented_adj(adata, npc=npc, NUM_AD_NEIGH=NUM_AD_NEIGH, neighbor_metric=neighbor_metric, graph_adj=graph_adj, use_us=use_us)
    else:
        if use_us:
            sc.pp.neighbors(adata, n_pcs=npc, n_neighbors=NUM_AD_NEIGH, use_rep='X_pca_combined', metric=neighbor_metric, key_added='adj')
        else:
            sc.pp.neighbors(adata, n_pcs=npc, n_neighbors=NUM_AD_NEIGH, use_rep='X_pca', metric=neighbor_metric, key_added='adj')
        
        binary_connectivities = adata.obsp['adj_' + graph_adj].copy()
        binary_connectivities.data = np.ones_like(binary_connectivities.data)
        binary_connectivities.setdiag(1)
        original_adjacency = binary_connectivities.toarray()

    # If use_spatial is True, calculate the spatial adjacency matrix using KNN
    if use_spatial and spatial_key in adata.obsm:
        spatial_coords = adata.obsm[spatial_key]
        
        # Calculate KNN-based spatial adjacency matrix
        knn = NearestNeighbors(n_neighbors=spatial_neighbors + 1)  # +1 because it includes the point itself
        knn.fit(spatial_coords)
        spatial_adj_matrix = knn.kneighbors_graph(spatial_coords)
        
        # Ensure diagonal elements are set to 1
        spatial_adj_matrix.setdiag(1)
        spatial_adjacency = spatial_adj_matrix.toarray()

        # Combine original and spatial adjacency matrices based on combine_mode
        if combine_mode == 'union':
            adjacency_matrix = np.logical_or(original_adjacency, spatial_adjacency).astype(int)
        elif combine_mode == 'overlap':
            # Calculate overlap
            adjacency_matrix = np.logical_and(original_adjacency, spatial_adjacency).astype(int)
            
            # Identify cells with no overlap
            no_overlap_cells = np.where(adjacency_matrix.sum(axis=1) == 1)[0]
            no_overlap_count = len(no_overlap_cells)
            print(f"Number of cells with no overlap: {no_overlap_count}")
            
            # For no-overlap cells, use the original adjacency
            for cell in no_overlap_cells:
                if spatial_first:
                    adjacency_matrix[cell] = spatial_adjacency[cell]
                else:
                    adjacency_matrix[cell] = original_adjacency[cell]
        else:
            raise ValueError("combine_mode must be 'union' or 'overlap'")
    else:
        adjacency_matrix = original_adjacency

    # Normalize 'unsplice' and 'splice'
    df['unsplice'] = df.groupby('gene_name')['unsplice'].transform(lambda x: x / x.max() if x.max() != 0 else x)
    df['splice'] = df.groupby('gene_name')['splice'].transform(lambda x: x / x.max() if x.max() != 0 else x)

    return df, adjacency_matrix, adata


def preprocess_anndata_fineT(df, adata, genes, npc = 30, NUM_AD_NEIGH = 30, SMOOTH_NEIGH = 100, use_us = False, adj_con = True, neighbor_metric='cosine', graph_adj ='connectivities'):

    # Clean Phase
    # Delete all elements in obsm except 'X_umap'
    keys_to_keep_obsm = ['X_tsne','X_umap', 'X_pre_embed', 'X_umap_pre_embed']
    keys_to_remove_obsm = [key for key in adata.obsm.keys() if key not in keys_to_keep_obsm]
    for key in keys_to_remove_obsm:
        del adata.obsm[key]

    # Delete all elements in layers except 'spliced', 'unspliced', and 'pred_cell_type'
    keys_to_keep_layers = ['spliced', 'unspliced', 'pred_cell_type']
    keys_to_remove_layers = [key for key in adata.layers.keys() if key not in keys_to_keep_layers]
    for key in keys_to_remove_layers:
        del adata.layers[key]
    
    del adata.uns['neighbors']
    
    # Delete all elements in obsp
    adata.obsp.clear()

    # Smooth Phase
    if use_us:
        adata = us_moments(adata=adata, npc=npc, n_neighbor=SMOOTH_NEIGH, neighbor_metric = neighbor_metric)
    else:
        sc.pp.neighbors(adata, n_pcs=npc, n_neighbors=SMOOTH_NEIGH, metric = neighbor_metric)
        scv.pp.moments(adata, n_pcs = npc, n_neighbors = SMOOTH_NEIGH)

    # Updata df Phase
    df, adata = update_moments_df(df, adata)
    
    # Construct Moments-based adjancency matrix
    if adj_con:
        adata, adjacency_matrix = construct_momented_adj(adata, npc = npc, NUM_AD_NEIGH = NUM_AD_NEIGH, neighbor_metric = neighbor_metric, graph_adj= graph_adj)
    else:
        adjacency_matrix, cell_name_to_id = adjancency_and_sequence(df, genes, NUM_AD_NEIGH = NUM_AD_NEIGH)
    
    # Normalize u and s
    df['unsplice'] =  df.groupby('gene_name')['unsplice'].transform(lambda x: x / x.max())
    df['splice'] =  df.groupby('gene_name')['splice'].transform(lambda x: x / x.max())
    
    return df, adjacency_matrix, adata

def fine_preprocess(adata, use_pretrain = True, npc = 30, NUM_AD_NEIGH = 30, SMOOTH_NEIGH = 100, use_us = False, adj_con = True, neighbor_metric='cosine', graph_adj ='connectivities'):

    if use_pretrain:
        return 
    # Smooth Phase
    if use_us:
        adata = us_moments(adata=adata, npc=npc, n_neighbor=SMOOTH_NEIGH, neighbor_metric = neighbor_metric)
    else:
        sc.pp.neighbors(adata, n_pcs=npc, n_neighbors=SMOOTH_NEIGH, metric = neighbor_metric)
        scv.pp.moments(adata, n_pcs = npc, n_neighbors = SMOOTH_NEIGH)

    # Updata df Phase
    df, adata = update_moments_df(df, adata)
    
    # Construct Moments-based adjancency matrix
    if adj_con:
        adata, adjacency_matrix = construct_momented_adj(adata, npc = npc, NUM_AD_NEIGH = NUM_AD_NEIGH, neighbor_metric = neighbor_metric, graph_adj= graph_adj)
    else:
        adjacency_matrix, cell_name_to_id = adjancency_and_sequence(df, genes, NUM_AD_NEIGH = NUM_AD_NEIGH)
    
    # Normalize u and s
    df['unsplice'] =  df.groupby('gene_name')['unsplice'].transform(lambda x: x / x.max())
    df['splice'] =  df.groupby('gene_name')['splice'].transform(lambda x: x / x.max())
    
    return df, adjacency_matrix, adata

def add_result_anndata(pretrain_df, embeddings_df, adata):
    # Add 'pred_cell_type', 'pred_cluster', 'pred_clus_weight' to adata.obs
    obs_update = pretrain_df[['cellID', 'pred_cell_type', 'pred_cluster', 'pred_clus_weight']].reset_index().drop_duplicates('cellID').set_index('cellID')
    adata.obs = adata.obs.merge(obs_update, left_index=True, right_index=True, how='left')

    # Function to reshape the pretrain_df data to fit the adata layers
    def reshape_data(layer_name):
        layer_data = pretrain_df.pivot(index='cellID', columns='gene_name', values=layer_name)
        return layer_data.reindex(index=adata.obs.index, columns=adata.var.index).values

    # Add 'recon_u', 'recon_s' as layers to adata
    adata.layers['recon_u'] = reshape_data('recon_u')
    adata.layers['recon_s'] = reshape_data('recon_s')

    # Add 'unsplice', 'splice' to layers and renamed as 'scaled_Mu' and 'scaled_Ms'
    adata.layers['scaled_Mu'] = reshape_data('unsplice')
    adata.layers['scaled_Ms'] = reshape_data('splice')

    if not embeddings_df.index.equals(adata.obs_names):
        raise ValueError("The index of embeddings_df does not match the obs in result_adata.")

    embeddings_df = embeddings_df.drop(columns=['time', 'predicted_cluster'])
    # Add the embeddings_df to result_adata under the obsm field with the key 'X_pre_embed'
    adata.obsm['X_pre_embed'] = embeddings_df.values

    # Calculate neighbors based on 'X_pre_embed' and store them in a new slot
    sc.pp.neighbors(adata, use_rep='X_pre_embed', key_added='pre_embed_neighbors')

    # Calculate UMAP based on the new neighbors and store it in a new slot
    temp_adata = sc.tl.umap(adata, neighbors_key='pre_embed_neighbors', copy = True)

    adata.obsm['X_umap_pre_embed'] = temp_adata.obsm['X_umap']
    del temp_adata
    del adata.obs['index']
    return adata

def add_prior_anndata(pretrain_df, adata):

    # Filter adata.var to only include genes present in pretrain_df
    filtered_genes = pretrain_df['gene_name'].unique()
    adata = adata[:, adata.var.index.isin(filtered_genes)].copy()
    # Add 'pred_cell_type', 'pred_cluster', 'pred_clus_weight' to adata.obs
    obs_update = pretrain_df[['cellID', 'random_cluster']].reset_index().drop_duplicates('cellID').set_index('cellID')
    adata.obs = adata.obs.merge(obs_update, left_index=True, right_index=True, how='left')

    # Add new variables 'pred_gene_type', 'Entropy_pred_cluster', 'Entropy_clusters', 'Entropy_random_cluster' to adata.var
    var_update = pretrain_df[['gene_name', 'pred_gene_type', 'Entropy_pred_cluster', 'Entropy_clusters', 'Entropy_random_cluster']].drop_duplicates('gene_name').set_index('gene_name')
    adata.var = adata.var.merge(var_update, left_index=True, right_index=True, how='left')

    # Function to reshape the pretrain_df data to fit the adata layers
    def reshape_data(layer_name):
        layer_data = pretrain_df.pivot(index='cellID', columns='gene_name', values=layer_name)
        return layer_data.reindex(index=adata.obs.index, columns=adata.var.index).values

    # Add 'pred_cell_type' as a layer to adata
    adata.layers['pred_cell_type'] = reshape_data('pred_cell_type')
    del adata.obs['index']

    return adata


def add_final_anndata(final_df, adata, embeddings_df):
    if adata.obs.keys().str.startswith('index').any():
        adata.obs = adata.obs.drop(columns=adata.obs.keys()[adata.obs.keys().str.startswith('index')])
    # Add 'pred_cell_type', 'pred_cluster', 'pred_clus_weight' to adata.obs
    obs_update = final_df[['cellID', 'pred_cluster_refine', 'pred_clus_weight_refine','pred_time']].reset_index().drop_duplicates('cellID').set_index('cellID')
    adata.obs = adata.obs.merge(obs_update, left_index=True, right_index=True, how='left')

    # Function to reshape the final_df data to fit the adata layers
    def reshape_data(layer_name):
        layer_data = final_df.pivot(index='cellID', columns='gene_name', values=layer_name)
        return layer_data.reindex(index=adata.obs.index, columns=adata.var.index).values

    # Add 'recon_u', 'recon_s' as layers to adata
    adata.layers['recon_u_refine'] = reshape_data('recon_u_refine')
    adata.layers['recon_s_refine'] = reshape_data('recon_s_refine')
    adata.layers['recon_alpha'] = reshape_data('recon_alpha')
    adata.layers['recon_beta'] = reshape_data('recon_beta')
    adata.layers['recon_gamma'] = reshape_data('recon_gamma')
    adata.layers['pred_vu'] = reshape_data('pred_vu')
    adata.layers['pred_vs'] = reshape_data('pred_vs')

    # Add 'unsplice', 'splice' to layers and renamed as 'scaled_Mu' and 'scaled_Ms'
    adata.layers['used_Mu'] = reshape_data('unsplice')
    adata.layers['used_Ms'] = reshape_data('splice')

    if not embeddings_df.index.equals(adata.obs_names):
        raise ValueError("The index of embeddings_df does not match the obs in result_adata.")

    embeddings_df = embeddings_df.drop(columns=['time', 'predicted_cluster'])
    # Add the embeddings_df to result_adata under the obsm field with the key 'X_pre_embed'
    adata.obsm['X_refine_embed'] = embeddings_df.values

    # Calculate neighbors based on 'X_pre_embed' and store them in a new slot
    sc.pp.neighbors(adata, use_rep='X_refine_embed', key_added='refine_embed_neighbors')

    # Calculate UMAP based on the new neighbors and store it in a new slot
    temp_adata = sc.tl.umap(adata, neighbors_key='refine_embed_neighbors', copy = True)

    adata.obsm['X_umap_refine_embed'] = temp_adata.obsm['X_umap']
    del temp_adata
    del adata.obs['index']
    return adata

def results_to_anndata(pretrain_df, obs_list, var_list, embedding_list, prior = False):
    # Get unique cell observations and gene variables
    cell_obs = pretrain_df[obs_list].drop_duplicates()
    gene_var = pretrain_df[var_list].drop_duplicates()

    # Get the list of pivot columns to avoid repeating the same operation
    if prior:
        pivot_columns = ['splice', 'unsplice', 'orig_splice', 'orig_unsplice', 'recon_s', 'recon_u', 'pred_cell_type']
    else:
        pivot_columns = ['splice', 'unsplice', 'orig_splice', 'orig_unsplice', 'recon_s', 'recon_u']
    pivot_results = {}

    # Pivot the dataframe to create matrices
    for column in pivot_columns:
        pivot_results[column] = pretrain_df.pivot(index='cellID', columns='gene_name', values=column).fillna(0)

    # Create AnnData object using the splice matrix
    adata = ad.AnnData(X=pivot_results['splice'].values, obs=cell_obs.set_index('cellID'), var=gene_var.set_index('gene_name'))

    # Add embeddings to AnnData object
    adata.obsm['embedding'] = adata.obs[embedding_list].values

    # Add other matrices as layers
    adata.layers['unspliced'] = pivot_results['unsplice'].values
    adata.layers['spliced'] = pivot_results['splice'].values
    adata.layers['orig_unsplice'] = pivot_results['orig_unsplice'].values
    adata.layers['orig_splice'] = pivot_results['orig_splice'].values
    adata.layers['recon_unsplice'] = pivot_results['recon_u'].values
    adata.layers['recon_splice'] = pivot_results['recon_s'].values
    if prior:
        adata.layers['pred_cell_type'] = pivot_results['pred_cell_type'].values

    return adata


class GeneExpressionDatasetAllGenes(Dataset):
    def __init__(self, df, gene_list):
        # Map pred_cell_type to numbers
        type_mapping = {'Up': 1, 'Auto': 0, 'Down': -1}
        df['pred_cell_type'] = df['pred_cell_type'].map(type_mapping)
        
        # Pivot to get unspliced, spliced, and pred_cell_type matrices
        unsplice_matrix = df.pivot_table(index='cellID', columns='gene_name', values='unsplice', aggfunc='first').fillna(0)
        splice_matrix = df.pivot_table(index='cellID', columns='gene_name', values='splice', aggfunc='first').fillna(0)
        type_matrix = df.pivot_table(index='cellID', columns='gene_name', values='pred_cell_type', aggfunc='first').fillna(0)
        orig_unsplice_matrix = df.pivot_table(index='cellID', columns='gene_name', values='orig_unsplice', aggfunc='first').fillna(0)
        orig_splice_matrix = df.pivot_table(index='cellID', columns='gene_name', values='orig_splice', aggfunc='first').fillna(0)
        
        # Ensure the columns are in a consistent order based on gene_list
        unsplice_matrix = unsplice_matrix[gene_list]
        splice_matrix = splice_matrix[gene_list]
        type_matrix = type_matrix[gene_list]
        orig_unsplice_matrix = orig_unsplice_matrix[gene_list]
        orig_splice_matrix = orig_splice_matrix[gene_list]

        # Concatenate unsplice and splice matrices along the columns
        features_matrix = pd.concat([unsplice_matrix, splice_matrix], axis=1)
        # Concatenate unsplice and splice matrices along the columns
        orig_features_matrix = pd.concat([orig_unsplice_matrix, orig_splice_matrix], axis=1)

        # Convert to tensor
        self.features = torch.tensor(features_matrix.values, dtype=torch.float32)
        self.orig_features = torch.tensor(orig_features_matrix.values, dtype=torch.float32)

        self.type_features = torch.tensor(type_matrix.values, dtype=torch.float32)
        
        # Cell IDs (as a tensor for possible use in learning, e.g., as part of the loss function)
        self.cell_ids = torch.tensor(features_matrix.index.values, dtype=torch.int64)
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Return both feature sets separately
        return self.features[idx], self.type_features[idx], self.cell_ids[idx], self.orig_features[idx]
    
class GeneExpressionDatasetAllGenes_adata(Dataset):
    def __init__(self, df, gene_list, adata):
        # Map pred_cell_type to numbers
        type_mapping = {'Up': 1, 'Auto': 0, 'Down': -1}
        df['pred_cell_type'] = df['pred_cell_type'].map(type_mapping)

        # Create a function to pivot and reindex the matrix
        def pivot_and_reindex(df, value, index, columns, fillna_value, gene_list, adata):
            matrix = df.pivot_table(index=index, columns=columns, values=value, aggfunc='first').fillna(fillna_value)
            matrix = matrix.reindex(adata.obs.index).fillna(0)
            matrix = matrix[gene_list].astype(np.float32)
            return matrix

        # Generate matrices
        unsplice_matrix = pivot_and_reindex(df, 'unsplice', 'cellID', 'gene_name', 0, gene_list, adata)
        splice_matrix = pivot_and_reindex(df, 'splice', 'cellID', 'gene_name', 0, gene_list, adata)
        type_matrix = pivot_and_reindex(df, 'pred_cell_type', 'cellID', 'gene_name', 0, gene_list, adata).astype(np.float32)
        orig_unsplice_matrix = pivot_and_reindex(df, 'orig_unsplice', 'cellID', 'gene_name', 0, gene_list, adata)
        orig_splice_matrix = pivot_and_reindex(df, 'orig_splice', 'cellID', 'gene_name', 0, gene_list, adata)

        # Concatenate unsplice and splice matrices along the columns
        features_matrix = np.concatenate([unsplice_matrix.values, splice_matrix.values], axis=1)
        orig_features_matrix = np.concatenate([orig_unsplice_matrix.values, orig_splice_matrix.values], axis=1)

        # Convert to tensor
        self.features = torch.tensor(features_matrix, dtype=torch.float32)
        self.orig_features = torch.tensor(orig_features_matrix, dtype=torch.float32)
        self.type_features = torch.tensor(type_matrix.values, dtype=torch.float32)

        # Cell IDs (as a NumPy array)
        self.cell_ids = unsplice_matrix.index.values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Return both feature sets separately
        return self.features[idx], self.type_features[idx], self.cell_ids[idx], self.orig_features[idx]


class GeneExpressionDataset_adata_velo(Dataset):
    def __init__(self, adata):
        # Map pred_cell_type to numbers
        type_mapping = {'Up': 1, 'Auto': 0, 'Down': -1}
        map_func = np.vectorize(type_mapping.get)  # Create a vectorized mapping function
        adata.layers['pred_cell_type'] = map_func(adata.layers['pred_cell_type'])  # Apply the mapping function to the layer data

        # Generate matrices
        unsplice_matrix = adata.layers['scale_Mu']
        splice_matrix = adata.layers['scale_Ms']
        type_matrix = adata.layers['pred_cell_type']
        orig_unsplice_matrix = adata.layers['unspliced']
        orig_splice_matrix = adata.layers['spliced']

        # Concatenate unsplice and splice matrices along the columns
        features_matrix = np.concatenate([unsplice_matrix, splice_matrix], axis=1)
        orig_features_matrix = np.concatenate([orig_unsplice_matrix.toarray(), orig_splice_matrix.toarray()], axis=1)

        # Convert to tensor
        self.features = torch.tensor(features_matrix, dtype=torch.float32)
        self.orig_features = torch.tensor(orig_features_matrix, dtype=torch.float32)
        self.type_features = torch.tensor(type_matrix, dtype=torch.float32)

        # Cell IDs (as a NumPy array)
        self.cell_ids = adata.obs.index.values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Return both feature sets separately
        return self.features[idx], self.type_features[idx], self.cell_ids[idx], self.orig_features[idx]

def preload_datasets_all_genes(df, genes, MODEL_MODE):
    # Extract relevant columns and convert to a more efficient format if not already done
    
    if MODEL_MODE == 'pretrain':
        df['pred_cell_type'] = 'Auto'

    relevant_df = df[['unsplice', 'splice', 'cellID', 'gene_name', 'pred_cell_type', 'orig_unsplice', 'orig_splice']].copy()
    dataset = GeneExpressionDatasetAllGenes(relevant_df, genes)

    return dataset

def preload_datasets_all_genes_anndata(df=None, MODEL_MODE='pretrain', adata=None):
    # Extract relevant columns and convert to a more efficient format if not already done
    if adata is None:
        raise ValueError("The 'adata' parameter must be provided.")
    if MODEL_MODE == 'pretrain':
        if df is None:
            raise ValueError("DataFrame 'df' must be provided in pretrain mode")
        df['pred_cell_type'] = 'Auto'
        relevant_df = df[['unsplice', 'splice', 'cellID', 'gene_name', 'pred_cell_type', 'orig_unsplice', 'orig_splice']].copy()
        genes = adata.var.index.tolist()
        dataset = GeneExpressionDatasetAllGenes_adata(relevant_df, genes, adata)
    else:
        if 'pred_cell_type' not in adata.layers:
            adata.layers['pred_cell_type'] = np.full((adata.shape[0], adata.shape[1]), 'Auto', dtype=object)
        # Compute the max values for Mu and Ms
        max_Mu = np.max(adata.layers['Mu'], axis=0)[np.newaxis, :]
        max_Ms = np.max(adata.layers['Ms'], axis=0)[np.newaxis, :]

        # Apply scaling only where the max value is non-zero
        adata.layers['scale_Mu'] = np.where(max_Mu > 0, adata.layers['Mu'] / max_Mu, 0)
        adata.layers['scale_Ms'] = np.where(max_Ms > 0, adata.layers['Ms'] / max_Ms, 0)

        # adata.layers['scale_Mu'] = adata.layers['Mu'] / np.max(adata.layers['Mu'], axis = 0)[np.newaxis, :]
        # adata.layers['scale_Ms'] = adata.layers['Ms'] / np.max(adata.layers['Ms'], axis = 0)[np.newaxis, :]
        dataset = GeneExpressionDataset_adata_velo(adata)

    return dataset

def normalize_adj(adj):
    # Compute the degree matrix
    deg = adj.sum(dim=1)
    
    # Compute D^-1/2
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0  # Handle division by zero
    
    # Create D^-1/2 matrix
    d_mat_inv_sqrt = torch.diag(deg_inv_sqrt)
    
    # Compute the normalized adjacency matrix
    normalized_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    
    return normalized_adj

def create_pyg_data(dataset, adjacency_matrix, normalize = False):
    edge_index, _ = dense_to_sparse(torch.tensor(adjacency_matrix, dtype=torch.float))
    PyG_data = Data(x = dataset.features,
                    edge_index = edge_index,
                    type_features=dataset.type_features,
                    orig_features=dataset.orig_features,
                    cell_ids=dataset.cell_ids)
    # PyG_data.adj = to_torch_coo_tensor(PyG_data.edge_index)
    PyG_data.adj = to_dense_adj(PyG_data.edge_index)[0]
    if normalize:
        PyG_data.adj = normalize_adj(PyG_data.adj)
    PyG_data.x = PyG_data.x.detach()
    return PyG_data

# Simulation 

class TimeDependentParameter:
    def __init__(self, end, start=None, rate=None):
        # Generate the actual end value from a uniform distribution around the mean 'end'
        self.mean_end = end
        self.end = np.random.uniform(0.9 * end, 1.1 * end)
        self.start = start if start is not None else self.end
        self.rate = rate if rate is not None else 0

    def value(self, t):
        if self.rate == 0:
            return self.end
        return self.end - (self.end - self.start) * np.exp(-self.rate * t)

def parse_group_params(params):
    parsed_params = []
    for group in params:
        alpha = TimeDependentParameter(*group['alpha'])
        beta = TimeDependentParameter(*group['beta'])
        gamma = TimeDependentParameter(*group['gamma'])
        parsed_params.append((alpha, beta, gamma))
    return parsed_params

def trans_dynamics(t, expr, alpha, beta, gamma):
    s, u = expr
    alpha_t = alpha.value(t)
    beta_t = beta.value(t)
    gamma_t = gamma.value(t)

    du_dt = alpha_t - beta_t * u
    ds_dt = beta_t * u - gamma_t * s
    return [ds_dt, du_dt]

def _generate_points(u0_start, s0_start, alpha, beta, gamma, t1, t2, samples):
    t_space = np.linspace(t1, t2, samples, endpoint=False)
    # t_space = np.floor(t_space/0.005)*0.005
    num_sol = solve_ivp(trans_dynamics, [t1, t2], [s0_start, u0_start], method='RK45', t_eval=t_space,
                        args=(alpha, beta, gamma))
    S, U = num_sol.y
    return U, S, t_space

def _jitter(U, S, scale):
    S += np.random.normal(loc=0.0, scale=scale * np.percentile(S, 99) / 20, size=len(S))
    U += np.random.normal(loc=0.0, scale=scale * np.percentile(U, 99) / 20, size=len(U))
    # S += np.random.normal(loc=0.0, scale=scale / 10, size=len(S))
    # U += np.random.normal(loc=0.0, scale=scale / 10, size=len(U))
    S, U = np.clip(S, 0, None), np.clip(U, 0, None)
    return U, S

# def simulate_groups(group_params, time_points, samples_per_group, mode, initial_conditions=(0, 0), noise_level=0.2):
#     results = []
#     all_U, all_S, all_t_space = [], [], []
#     current_initial_conditions = initial_conditions
    
#     for i, (alpha, beta, gamma) in enumerate(group_params):
#         t1, t2 = time_points[i]
#         if mode == 'sequential' and i > 0:
#             current_initial_conditions = (S[-1], U[-1])
#         samples = samples_per_group[i]
#         U, S, t_space = _generate_points(current_initial_conditions[1], current_initial_conditions[0], 
#                                          alpha, beta, gamma, t1, t2, samples)
#         all_U.extend(U)
#         all_S.extend(S)
#         all_t_space.extend(t_space.round(5))
        
#     # Apply jitter outside the loop
#     all_U, all_S = _jitter(all_U, all_S, noise_level)
    
#     # Reconstruct results with jittered data
#     index = 0
#     for i, (alpha, beta, gamma) in enumerate(group_params):
#         samples = samples_per_group[i]
#         for j in range(samples):
#             results.append({
#                 'time': all_t_space[index],
#                 'unsplice': all_U[index],
#                 'splice': all_S[index],
#                 'cellID': len(results) + 1,
#                 'group_type': mode,
#                 'alpha': alpha.value(all_t_space[index]),
#                 'beta': beta.value(all_t_space[index]),
#                 'gamma': gamma.value(all_t_space[index]),
#                 'group_index': i
#             })
#             index += 1

#     return pd.DataFrame(results)

def simulate_groups(group_params, time_points, samples_per_group, mode, initial_conditions=(0, 0), noise_level=0.2):
    results = []
    all_U, all_S, all_t_space = [], [], []
    current_initial_conditions = initial_conditions
    
    first_group_end_point = None
    
    for i, (alpha, beta, gamma) in enumerate(group_params):
        t1, t2 = time_points[i]
        if mode == 'sequential' and i > 0:
            current_initial_conditions = (S[-1], U[-1])
        elif mode == 'progenitor' and i > 0:
            current_initial_conditions = first_group_end_point
        
        samples = samples_per_group[i]
        U, S, t_space = _generate_points(current_initial_conditions[1], current_initial_conditions[0], 
                                         alpha, beta, gamma, t1, t2, samples)
        
        if i == 0:
            first_group_end_point = (S[-1], U[-1])
        
        all_U.extend(U)
        all_S.extend(S)
        all_t_space.extend(t_space.round(5))
        
    # Apply jitter outside the loop
    all_U, all_S = _jitter(all_U, all_S, noise_level)
    
    # Reconstruct results with jittered data
    index = 0
    for i, (alpha, beta, gamma) in enumerate(group_params):
        samples = samples_per_group[i]
        for j in range(samples):
            results.append({
                'time': all_t_space[index],
                'unsplice': all_U[index],
                'splice': all_S[index],
                'cellID': len(results) + 1,
                'group_type': mode,
                'alpha': alpha.value(all_t_space[index]),
                'beta': beta.value(all_t_space[index]),
                'gamma': gamma.value(all_t_space[index]),
                'group_index': i
            })
            index += 1
    
    return pd.DataFrame(results)

def run_simulations(num_genes, group_params, time_points, samples_per_group, mode, noise_level=0.2, combined = False, NUM_GROUP = None):
    all_data = pd.DataFrame()
    # parsed_group_params = parse_group_params(group_params)
    
    for gene_index in range(num_genes):
        parsed_group_params = parse_group_params(group_params)
        if combined:
            if NUM_GROUP is not None:
                gene_name = f"Gene_{num_genes * (NUM_GROUP - 1) + 1 + gene_index}"
            else:
                gene_name = f"Gene_{num_genes * (len(group_params) - 1) + 1 + gene_index}"
        else:
            gene_name = f"Gene_{gene_index+1}"
        df_gene = simulate_groups(parsed_group_params, time_points, samples_per_group, mode, noise_level=noise_level)
        df_gene['gene_name'] = gene_name
        all_data = pd.concat([all_data, df_gene], ignore_index=True)
    
    return all_data

def us_transition_matrix(
    adata, 
    velocity_u_key='pred_vu', 
    velocity_s_key='pred_vs',
    unspliced_key='used_Mu', 
    spliced_key='used_Ms'
):
    spliced = adata.layers[spliced_key].toarray() if hasattr(adata.layers[spliced_key], 'toarray') else adata.layers[spliced_key]
    unspliced = adata.layers[unspliced_key].toarray() if hasattr(adata.layers[unspliced_key], 'toarray') else adata.layers[unspliced_key]

    # Combine 'spliced' and 'unspliced' layers by concatenating along the feature axis (genes)
    combined = np.concatenate([unspliced, spliced], axis=1)

    # Create a new AnnData object with the combined data
    combined_adata = sc.AnnData(X=combined)
    combined_adata.obs.index = adata.obs.index

    combined_adata.layers['velocity'] = np.concatenate((adata.layers[velocity_u_key], adata.layers[velocity_s_key]),axis=1)
    combined_adata.layers['used_Mu_Ms'] = combined_adata.X
    combined_adata.obsm = adata.obsm
    combined_adata.obs = adata.obs
    
    return combined_adata

def normalize_l2_anndata(adata, layer_vu='pred_vu', layer_vs='pred_vs'):
    # Extract the layers
    pred_vu = adata.layers[layer_vu]
    pred_vs = adata.layers[layer_vs]
    
    # Calculate the L2 norm for each pair
    norms = np.sqrt(pred_vu**2 + pred_vs**2)
    
    # Avoid division by zero by setting zero norms to 1 (since 0 vector stays 0)
    norms = np.where(norms == 0, 1, norms)
    
    # Normalize each pair
    norm_vu = pred_vu / norms
    norm_vs = pred_vs / norms
    
    # Update the AnnData object with normalized values
    adata.layers['pred_vu_norm'] = norm_vu
    adata.layers['pred_vs_norm'] = norm_vs

    return adata



def compute_gene_specific_adj(velo_adata, n_neighbors, device):

    cells, genes = velo_adata.n_obs, velo_adata.n_vars
    neighbors_indices = torch.zeros((cells, genes, n_neighbors), dtype=torch.long, device = device)

    # Split unsplice and splice into separate matrices for easier manipulation
    unsplice = torch.tensor(velo_adata.layers['scale_Mu'], device = device)
    splice = torch.tensor(velo_adata.layers['scale_Ms'], device = device)

    # Process each gene to find neighbors
    for gene_idx in range(genes):
        # Combine unsplice and splice values for the current gene into a single matrix
        gene_data = torch.stack((unsplice[:, gene_idx], splice[:, gene_idx]), dim=1)

        # Compute pairwise Euclidean distance
        dist_matrix = torch.cdist(gene_data, gene_data, p=2)

        # Find indices of the nearest neighbors (excluding self)
        # Sort distances and get indices
        sorted_indices = torch.argsort(dist_matrix, dim=1)

        # Exclude self (first column) and select the n_neighbors
        neighbors_indices[:, gene_idx, :] = sorted_indices[:, 1:n_neighbors + 1]

    adjacency_matrices = []

    # Create source indices for all cells
    source_indices = torch.arange(cells, device = device).unsqueeze(1).repeat(1, n_neighbors).flatten()

    for gene_idx in range(genes):
        # Extract the target indices (neighbors) for the current gene
        target_indices = neighbors_indices[:, gene_idx, :].flatten()

        # Create the adjacency matrix for the current gene
        adjacency_matrix = torch.stack([source_indices, target_indices], dim=0)
        adjacency_matrices.append(adjacency_matrix)

    return adjacency_matrices

def df_2_anndata(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    # Specify obs and var columns
    obs_columns = ['cellID',  'lineage_cluster', 'time']
    var_columns = ['Gene_pattern']
    layer_columns = ['unsplice', 'splice', 'alpha', 'beta', 'gamma', 'ture_vu_normbeta', 'ture_vs_normbeta', 'ture_vu', 'ture_vs','clusters','group_index']
    obsm_columns = ['embedding1', 'embedding2']

    # Create obs and var dataframes
    obs_df = df[obs_columns].drop_duplicates().set_index('cellID')
    var_df = df[['gene_name'] + var_columns].drop_duplicates().set_index('gene_name')

    # Ensure obs_df and var_df indices are strings
    obs_df.index = obs_df.index.astype(str)
    var_df.index = var_df.index.astype(str)

    # Create the layers dictionary
    layers = {col: df.pivot(index='cellID', columns='gene_name', values=col) for col in layer_columns}

    # Ensure layers indices are strings
    for key in layers.keys():
        layers[key].index = layers[key].index.astype(str)
        layers[key].columns = layers[key].columns.astype(str)
        layers[key] = layers[key].reindex(index=obs_df.index, columns=var_df.index, fill_value=0)

    # Create the obsm dictionary
    obsm = {col: df.pivot(index='cellID', columns='gene_name', values=col) for col in obsm_columns}

    # Ensure obsm indices are strings
    for key in obsm.keys():
        obsm[key].index = obsm[key].index.astype(str)
        obsm[key].columns = obsm[key].columns.astype(str)
        obsm[key] = obsm[key].reindex(index=obs_df.index, columns=var_df.index, fill_value=0)

    # Create the AnnData object
    adata = ad.AnnData(
        X=np.zeros((obs_df.shape[0], var_df.shape[0])),  # Placeholder for the X matrix
        obs=obs_df,
        var=var_df,
        layers=layers,
        obsm=obsm
    )
    # Convert unsplice and splice layers to sparse format
    adata.layers['unspliced'] = csr_matrix(adata.layers.pop('unsplice'))
    adata.layers['spliced'] = csr_matrix(adata.layers.pop('splice'))
    return adata

def downsample_adata_by_celltype(adata, celltype_column, fraction):
    sampled_indices = []
    for celltype in adata.obs[celltype_column].unique():
        celltype_indices = adata.obs[adata.obs[celltype_column] == celltype].index
        sample_size = int(len(celltype_indices) * fraction)
        sampled_indices.extend(np.random.choice(celltype_indices, sample_size, replace=False))
    
    return adata[sampled_indices].copy()

def downsample_adata_randomly(adata, fraction, seed=618):
    np.random.seed(seed)
    # Calculate total sample size
    total_cells = len(adata.obs)
    sample_size = int(total_cells * fraction)

    # Randomly sample indices from the whole dataset
    sampled_indices = np.random.choice(adata.obs.index, sample_size, replace=False)
    
    # Return the sampled adata object
    return adata[sampled_indices].copy()

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='X_pre_embed', random_seed=618):
   
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['pred_cluster'] = mclust_res
    adata.obs['pred_cluster'] = adata.obs['pred_cluster'].astype('int')
    adata.obs['pred_cluster'] = adata.obs['pred_cluster'].astype('category')
    return adata

def clean_anndata(adata):
    # Renaming obs columns
    obs_rename_dict = {
        'pred_cluster': 'pretrain_cluster',
        'pred_cluster_refine': 'Expert',
        'pred_clus_weight_refine': 'Expert Weight',
        'pred_time': 'Pred Time'
    }
    for old_name, new_name in obs_rename_dict.items():
        if old_name in adata.obs:
            adata.obs.rename(columns={old_name: new_name}, inplace=True)
    
    # Renaming var columns
    var_rename_dict = {
        'Entropy_pred_cluster': 'Entropy_pretrain_cluster'
    }
    for old_name, new_name in var_rename_dict.items():
        if old_name in adata.var:
            adata.var.rename(columns={old_name: new_name}, inplace=True)
    
    # Renaming layers
    layers_rename_dict = {
        'pred_cell_type': 'init_regulate_state',
        'pred_cell_type_refine': 'regulate_state',
        'recon_s': 'pretrain_recon_s',
        'recon_u': 'pretrain_recon_u',
        'recon_s_refine': 'final_recon_s',
        'recon_u_refine': 'final_recon_u',
        'scale_Ms_refine': 'model_Ms',
        'scale_Mu_refine': 'model_Mu',
        'spliced': 'orig_s',
        'unspliced': 'orig_u'
    }
    
    for old_name, new_name in layers_rename_dict.items():
        if old_name in adata.layers:
            adata.layers[new_name] = adata.layers.pop(old_name)

    # Deleting specific layers
    layers_to_delete = ['Ms', 'Mu', 'ambiguous', 'confidence', 'ds', 'du', 'fs', 'fu', 'matrix','pred_group_type', 'ps', 'pu']
    for layer in layers_to_delete:
        if layer in adata.layers:
            del adata.layers[layer]

    # Adding 'Pred Time' to layers
    if 'Pred Time' in adata.obs:
        # Get the 'Pred Time' column from obs
        pred_time_values = adata.obs['Pred Time'].values
        
        # Create a new layer where each cell has the same 'Pred Time' value for all genes
        # The shape of the new layer should match adata.X (n_obs, n_vars)
        adata.layers['pred_time_layer'] = np.tile(pred_time_values[:, np.newaxis], (1, adata.n_vars))
    # Copy recon_alpha, recon_beta, recon_gamma to obsm
    adata.obsm['X_alpha'] = adata.layers['recon_alpha']
    adata.obsm['X_beta'] = adata.layers['recon_beta']
    adata.obsm['X_gamma'] = adata.layers['recon_gamma']
    
    # Concatenate recon_alpha, recon_beta, recon_gamma to form X_para
    adata.obsm['X_para'] = np.concatenate(
        [adata.layers['recon_alpha'], 
         adata.layers['recon_beta'], 
         adata.layers['recon_gamma']], 
        axis=1
    )
    
    # Concatenate recon_alpha, recon_beta, recon_gamma, and pred_time_layer to form X_para_t
    if 'pred_time_layer' in adata.layers:
        adata.obsm['X_para_t'] = np.concatenate(
            [adata.layers['recon_alpha'], 
             adata.layers['recon_beta'], 
             adata.layers['recon_gamma'],
             adata.obs['Pred Time'].values.reshape(-1, 1)], 
            axis=1
        )
        adata.obsm['X_refine_embed_t'] = np.concatenate(
            [adata.obsm['X_refine_embed'],
             adata.obs['Pred Time'].values.reshape(-1, 1)], 
            axis=1
        )
    
    adata.layers['recon_alpha_norm'] = adata.layers['recon_alpha'] / adata.layers['recon_beta']
    adata.layers['recon_gamma_norm'] = adata.layers['recon_gamma'] / adata.layers['recon_beta']
    adata.obs['Expert'] = adata.obs['Expert'].astype(str)
    return adata

def to_dynamo_format(vdata):
    vdata_new = vdata.copy()
    vdata_new.layers['X_spliced'] = vdata_new.layers.pop('orig_s')
    vdata_new.layers['X_unspliced'] = vdata_new.layers.pop('orig_u')
    vdata_new.layers['M_s'] = vdata_new.layers.pop('model_Ms')
    vdata_new.layers['M_u'] = vdata_new.layers.pop('model_Mu')
    vdata_new.layers['alpha'] = vdata_new.layers.pop('recon_alpha')
    vdata_new.layers['beta'] = vdata_new.layers.pop('recon_beta')
    vdata_new.layers['gamma'] = vdata_new.layers.pop('recon_gamma')
    vdata_new.layers['velocity_U'] = vdata_new.layers.pop('pred_vu_norm')
    vdata_new.layers['velocity_S'] = vdata_new.layers.pop('pred_vs_norm')
    vdata_new.var['use_for_transition'] = True
    vdata_new.var['use_for_dynamics'] = True
    vdata_new.uns['dynamics']={'filter_gene_mode': 'final', 't': None, 'group': None, 'X_data': None, 'X_fit_data': None, 'asspt_mRNA': 'ss', 'experiment_type': 'conventional', 'normalized': True, 'model': 'static', 'est_method': 'ols', 'has_splicing': True, 'has_labeling': False, 'splicing_labeling': False, 'has_protein': False, 'use_smoothed': True, 'NTR_vel': False, 'log_unnormalized': True, 'fraction_for_deg': False}
    return vdata_new

# Function to calculate cosine similarities for a given layer
def calculate_gene_cosine_similarity(total_layer, miss_layer):
    cosine_similarities = []
    for i in range(total_layer.shape[1]):
        total_gene = total_layer[:, i].reshape(1, -1)
        miss_gene = miss_layer[:, i].reshape(1, -1)
        cosine_sim = cosine_similarity(total_gene, miss_gene)[0, 0]
        cosine_similarities.append(cosine_sim)
    return cosine_similarities

def keep_type(adata, nodes, target, k_cluster):
    """Select cells of targeted type
    
    Args:
        adata (Anndata): 
            Anndata object.
        nodes (list): 
            Indexes for cells
        target (str): 
            Cluster name.
        k_cluster (str): 
            Cluster key in adata.obs dataframe

    Returns:
        list: 
            Selected cells.

    """
    return nodes[adata.obs[k_cluster][nodes].values == target]
def cross_boundary_correctness(
    adata, 
    k_cluster, 
    k_velocity, 
    cluster_edges, 
    return_raw=False, 
    x_emb="X_umap"
):
    """Cross-Boundary Direction Correctness Score (A->B)
    
    Args:
        adata (Anndata): 
            Anndata object.
        k_cluster (str): 
            key to the cluster column in adata.obs DataFrame.
        k_velocity (str): 
            key to the velocity matrix in adata.obsm.
        cluster_edges (list of tuples("A", "B")): 
            pairs of clusters has transition direction A->B
        return_raw (bool): 
            return aggregated or raw scores.
        x_emb (str): 
            key to x embedding for visualization.
        
    Returns:
        dict: 
            all_scores indexed by cluster_edges or mean scores indexed by cluster_edges
        float: 
            averaged score over all cells.
    """
    scores = {}
    all_scores = {}
    
    # Get the velocity embedding and x embedding
    if x_emb == "X_umap":
        v_emb = adata.obsm['{}'.format(k_velocity)]
    else:
        v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(k_velocity)][0]]
        
    x_emb = adata.obsm[x_emb]
    
    # Retrieve connectivity matrix
    connectivities = adata.obsp['connectivities']
    
    for u, v in cluster_edges:
        sel = adata.obs[k_cluster] == u
        selected_indices = np.where(sel)[0]
        
        type_score = []

        # Iterate through selected cells
        for idx in selected_indices:
            # Get neighbors using connectivities matrix
            neighbors = connectivities[idx].nonzero()[1]
            
            # Keep neighbors of the target cluster type
            boundary_nodes = keep_type(adata, neighbors, v, k_cluster)
            
            # Get position and velocity of current cell
            x_pos = x_emb[idx]
            x_vel = v_emb[idx]
            
            if len(boundary_nodes) == 0:
                continue
            
            # Calculate direction similarity
            position_dif = x_emb[boundary_nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1, -1)).flatten()
            type_score.append(np.mean(dir_scores))
        
        # Store the scores for this transition (u -> v)
        scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score
    
    if return_raw:
        return all_scores
    
    return scores, np.mean([sc for sc in scores.values()])

def TimeNorm(result_adata, key='Pred Time'):
    pred_time = result_adata.obs[key]
    result_adata.obs[key] = (pred_time - pred_time.min()) / (pred_time.max() - pred_time.min())
    return result_adata

def plot_pred_time_correlation(adata1, adata2, timekey='Pred Time', xlab='Predict time using full data', ylab='Predict time using anchor data', title='Comparison of time for each cell', save_path=None):
    """
    Plot regression and compute Spearman and Pearson correlation coefficients
    between 'Pred Time' of two AnnData objects.
    """
    # Normalize 'Pred Time'
    for adata in [adata1, adata2]:
        adata.obs[timekey] = (
            adata.obs[timekey] - adata.obs[timekey].min()
        ) / (adata.obs[timekey].max() - adata.obs[timekey].min())
    
    # Compute correlations
    spearman_corr, _ = spearmanr(adata1.obs[timekey], adata2.obs[timekey])
    pearson_corr, _ = pearsonr(adata1.obs[timekey], adata2.obs[timekey])
    
    # Plot regression
    plt.figure(figsize=(4, 3))
    sns.regplot(
        x=adata1.obs[timekey],
        y=adata2.obs[timekey],
        scatter_kws={'s': 2, 'edgecolor': None},
        line_kws={'color': 'r', 'linewidth': 2}
    )
    plt.xlabel(xlab, fontsize=7)
    plt.ylabel(ylab, fontsize=7)
    plt.title(title, fontsize=7)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.text(0, 0.98, f'Spearman Correlation: {spearman_corr:.4f}', fontsize=7)
    plt.text(0, 0.91, f'Pearson Correlation: {pearson_corr:.4f}', fontsize=7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
    else:
        plt.show()
    
    return spearman_corr, pearson_corr

def calculate_layers_cosine_similarity(adata1, adata2, layers, save_path=None):
    """
    Calculate cosine similarities for specified layers between two AnnData objects.
    Plot histograms for each layer.
    """
    # Ensure overlapping genes
    overlap_genes = adata1.var_names.intersection(adata2.var_names)
    adata1 = adata1[:, overlap_genes].copy()
    adata2 = adata2[:, overlap_genes].copy()
    
    all_cosine_data = []
    
    for layer_name in layers:
        layer1 = adata1.layers[layer_name]
        layer2 = adata2.layers[layer_name]
        
        # Transpose to (genes x cells)
        layer1 = layer1.T
        layer2 = layer2.T
        
        cosine_similarities = cosine_similarity(layer1, layer2).diagonal()
        all_cosine_data.extend([
            {'Layer': layer_name, 'Cosine Similarity': cos_sim}
            for cos_sim in cosine_similarities
        ])
    
    # Create DataFrame
    cosine_df = pd.DataFrame(all_cosine_data)
    
    # Plot histograms
    fig, axes = plt.subplots(1, len(layers), figsize=(2.5*len(layers), 3))
    if len(layers) == 1:
        axes = [axes]
    for i, layer_name in enumerate(layers):
        sns.histplot(
            data=cosine_df[cosine_df['Layer'] == layer_name],
            x='Cosine Similarity',
            bins=np.linspace(0, 1, 50),
            kde=True,
            fill=True,
            ax=axes[i],
            alpha=0.5
        )
        axes[i].set_title(f'Cosine Similarity of {layer_name}', fontsize=7)
        axes[i].set_xlim(0, 1)
        axes[i].tick_params(axis='both', labelsize=7)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
    else:
        plt.show()

def compare_velocity(adata1, adata2, save_path=None):
    """
    Compute cosine similarity between velocity_umap embeddings of two AnnData objects.
    Plot histogram.
    """
    velocity_umap_similarity = cosine_similarity(
        adata1.obsm['velocity_umap'],
        adata2.obsm['velocity_umap']
    ).diagonal()
    
    # Plot histogram
    plt.figure(figsize=(2.5, 3))
    sns.histplot(
        velocity_umap_similarity,
        bins=np.linspace(0, 1, 50),
        alpha=0.7
    )
    plt.title("Cosine Similarity for Velocity")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
    else:
        plt.show()
    
    return velocity_umap_similarity

def cross_boundary_correctness_multiple(adata_list, labels, cluster_edges, save_path=None):
    """
    Compute cross-boundary correctness scores for multiple datasets.
    """
    cross_boundary_scores = {edge: [] for edge in cluster_edges}
    
    for adata in adata_list:
        result = cross_boundary_correctness(
            adata,
            k_cluster='celltype',
            k_velocity='velocity_umap',
            cluster_edges=cluster_edges,
            x_emb='X_umap'
        )
        for edge, score in result[0].items():
            cross_boundary_scores[edge].append(score)
    
    # Prepare and plot heatmap
    heatmap_data = np.array([cross_boundary_scores[edge] for edge in cluster_edges])
    plt.figure(figsize=(len(labels)*0.8, 3))
    sns.heatmap(
        heatmap_data,
        annot=True,
        cmap='Blues',
        alpha=0.7,
        xticklabels=labels,
        yticklabels=[f'{edge[0]} -> {edge[1]}' for edge in cluster_edges]
    )
    plt.title("Cross-boundary Correctness Scores across Different Datasets")
    plt.xlabel("Dataset")
    plt.ylabel("Cell Type Transition")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    
def compare_velocity_multiple(adata_reference, adata_list, labels, save_path=None):
    """
    Compute cosine similarity of velocity_umap embeddings between a reference dataset
    and multiple other datasets.
    """
    cosine_similarities = []
    for adata in adata_list:
        similarity = cosine_similarity(
            adata_reference.obsm['velocity_umap'],
            adata.obsm['velocity_umap']
        ).diagonal()
        cosine_similarities.append(similarity)
    
    # Plot violin plots
    plt.figure(figsize=(len(labels)*0.6, 2))
    sns.violinplot(
        data=cosine_similarities,
        palette='viridis'
    )
    plt.xticks(ticks=np.arange(len(labels)), labels=labels)
    plt.title("Cosine Similarity for Velocity across Datasets")
    plt.xlabel("Missing Dataset")
    plt.ylabel("Cosine Similarity")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    
    return cosine_similarities

def compare_velocity_umap(adata, sub_key, parts, colors_list = ["#FFED6F", "#FB8072", "#80B1D3", '#1F78B4', "#8DD3C7"], show = True, result_path='', tail='velocity_comparison_all_parts.pdf',rotation='True',title='Velocity Comparasion between Global and Expert',size=(3,2)):
    """
    Compare the velocity UMAP between full data and subsets for given parts.

    Parameters:
    - adata: AnnData object containing the full dataset.
    - sub_key: str, the key in adata.obs to subset on (e.g., 'Expert').
    - parts: list, values of sub_key to process (e.g., ['3', '4']).
    - result_path: str, optional path to save the results.

    Returns:
    None
    """
    # Ensure the result path ends with '/'
    if result_path and not result_path.endswith('/'):
        result_path += '/'

    # Get the 'Expert' labels from adata
    expert_labels = adata.obs[sub_key].cat.categories.tolist()
    expert_color_dict = dict(zip(expert_labels, colors_list))

    # Set the colors in adata.uns
    adata.uns[f'{sub_key}_colors'] = [expert_color_dict[expert] for expert in expert_labels]

    # Process full data
    combined_adata_full = us_transition_matrix(
        adata.copy(),
        velocity_u_key='pred_vu',
        velocity_s_key='pred_vs',
        unspliced_key='model_Mu',
        spliced_key='model_Ms'
    )
    velocity_graph(
        combined_adata_full,
        vkey='velocity',
        xkey='used_Mu_Ms',
        basis='X_umap',
        show_progress_bar=False
    )

    # Compute and store the velocity embeddings
    scv.tl.velocity_embedding(combined_adata_full, basis='umap')
    test_full = combined_adata_full.copy()

    # Set the colors in test_full
    test_full.uns[f'{sub_key}_colors'] = [expert_color_dict[expert] for expert in expert_labels]

    # Initialize lists to collect similarities and labels
    all_similarities = []
    all_labels = []

    # Create a figure with 2 rows and as many columns as there are parts
    num_parts = len(parts)
    fig, axes = plt.subplots(nrows=2, ncols=num_parts, figsize=(2 * num_parts, 4))

    # Loop over each part in parts
    for i, part in enumerate(parts):
        # Subset full data to the current part
        test_full_part = test_full[test_full.obs[sub_key] == part].copy()
        velocity_umap_full = test_full_part.obsm['velocity_umap']

        # Process sub data for the current part
        adata_sub = adata[adata.obs[sub_key] == part].copy()
        # Set the colors in adata_sub
        adata_sub.uns[f'{sub_key}_colors'] = [expert_color_dict[expert] for expert in expert_labels]

        combined_adata_sub = us_transition_matrix(
            adata_sub,
            velocity_u_key='pred_vu',
            velocity_s_key='pred_vs',
            unspliced_key='model_Mu',
            spliced_key='model_Ms'
        )
        velocity_graph(
            combined_adata_sub,
            vkey='velocity',
            xkey='used_Mu_Ms',
            basis='X_umap',
            show_progress_bar=False
        )

        # Compute and store the velocity embeddings
        scv.tl.velocity_embedding(combined_adata_sub, basis='umap')
        test_sub = combined_adata_sub.copy()

        # Set the colors in test_sub
        test_sub.uns[f'{sub_key}_colors'] = [expert_color_dict[expert] for expert in expert_labels]

        # Plot velocity embedding stream for full data's corresponding part
        scv.pl.velocity_embedding_stream(
            test_full,
            basis='X_umap',
            color=sub_key,
            groups=part,
            ax=axes[0, i],
            show=False,
            min_mass=4,
            density=1,
            arrow_size=0.7,
            linewidth=0.7,
            xlabel='UMAP1',
            ylabel='UMAP2',
            vkey='velocity',
            title=f'Full Data\n{sub_key} {part}'
        )

        # Record the x and y limits
        xlims = axes[0, i].get_xlim()
        ylims = axes[0, i].get_ylim()

        # Plot velocity embedding stream for sub data
        scv.pl.velocity_embedding_stream(
            test_sub,
            basis='X_umap',
            color=sub_key,
            ax=axes[1, i],
            show=False,
            min_mass=4,
            density=1,
            arrow_size=0.7,
            linewidth=0.7,
            xlabel='UMAP1',
            ylabel='UMAP2',
            vkey='velocity',
            title=f'Subset Data\n{sub_key} {part}'
        )

        # Set the x and y limits to match the full data's plot
        axes[1, i].set_xlim(xlims)
        axes[1, i].set_ylim(ylims)

        # Remove y-axis labels for inner plots to reduce clutter
        if i > 0:
            axes[0, i].set_ylabel('')
            axes[1, i].set_ylabel('')

        # Align the cells by their unique identifiers
        # Get the intersection of cell names
        common_cells = test_full_part.obs_names.intersection(test_sub.obs_names)

        # Ensure the cell ordering is the same in both datasets
        velocity_umap_full_aligned = pd.DataFrame(
            velocity_umap_full,
            index=test_full_part.obs_names
        ).loc[common_cells].values

        velocity_umap_sub_aligned = pd.DataFrame(
            test_sub.obsm['velocity_umap'],
            index=test_sub.obs_names
        ).loc[common_cells].values

        # Compute cosine similarity between aligned cells
        velocity_umap_similarity = cosine_similarity(
            velocity_umap_full_aligned, velocity_umap_sub_aligned
        ).diagonal()

        # Collect similarities and labels
        all_similarities.extend(velocity_umap_similarity)
        all_labels.extend([part] * len(velocity_umap_similarity))

    # Adjust layout and save the combined figure
    plt.tight_layout()
    if show:
        plt.savefig(f'{result_path}_stream_{tail}', bbox_inches='tight')
        plt.show()
    else:
        plt.show()
    plt.close(fig)

    # Create a DataFrame for plotting
    df_similarity = pd.DataFrame({
        'Cosine Similarity': all_similarities,
        sub_key: all_labels
    })

    # Create a palette for the violin plot
    palette = [expert_color_dict[part] for part in parts]

    # Plot violin plot for all parts
    plt.figure(figsize=size)
    sns.violinplot(
        x=sub_key,
        y='Cosine Similarity',
        data=df_similarity,
        inner='box',
        palette=palette
    )
    plt.title(title)
    plt.xlabel(sub_key)
    if rotation:
        plt.xticks(rotation=90)
    plt.ylabel("Cosine Similarity")
    plt.tight_layout()
    if show:
        plt.savefig(f'{result_path}_violin_{tail}', bbox_inches='tight')
        plt.show()
    else:
        plt.show()
    plt.close()
        


def compare_velocity_umap_other(adata, sub_key, parts, colors_list = ["#FFED6F", "#FB8072", "#80B1D3", '#1F78B4', "#8DD3C7"],size=(3,4), show = True, result_path='', save_name='scvelo.pdf'):
    """
    Compare the velocity UMAP between full data and subsets for given parts.

    Parameters:
    - adata: AnnData object containing the full dataset.
    - sub_key: str, the key in adata.obs to subset on (e.g., 'Expert').
    - parts: list, values of sub_key to process (e.g., ['3', '4']).
    - result_path: str, optional path to save the results.

    Returns:
    None
    """
    # Ensure the result path ends with '/'
    if result_path and not result_path.endswith('/'):
        result_path += '/'

    # Get the 'Expert' labels from adata
    expert_labels = adata.obs[sub_key].cat.categories.tolist()
    expert_color_dict = dict(zip(expert_labels, colors_list))

    # Set the colors in adata.uns
    adata.uns[f'{sub_key}_colors'] = [expert_color_dict[expert] for expert in expert_labels]
    scv.tl.velocity_graph(adata)
    # Compute and store the velocity embeddings
    scv.tl.velocity_embedding(adata, basis='umap')
    test_full = adata.copy()

    # Set the colors in test_full
    test_full.uns[f'{sub_key}_colors'] = [expert_color_dict[expert] for expert in expert_labels]

    # Initialize lists to collect similarities and labels
    all_similarities = []
    all_labels = []

    # Create a figure with 2 rows and as many columns as there are parts
    num_parts = len(parts)
    fig, axes = plt.subplots(nrows=2, ncols=num_parts, figsize=(2 * num_parts, 4))

    # Loop over each part in parts
    for i, part in enumerate(parts):
        # Subset full data to the current part
        test_full_part = test_full[test_full.obs[sub_key] == part].copy()
        velocity_umap_full = test_full_part.obsm['velocity_umap']

        # Process sub data for the current part
        adata_sub = adata[adata.obs[sub_key] == part].copy()
        # Set the colors in adata_sub
        adata_sub.uns[f'{sub_key}_colors'] = [expert_color_dict[expert] for expert in expert_labels]
        try:
            scv.tl.velocity_graph(adata_sub)
        except Exception as e:
            # If the neighbors can't use in subdata, it will introduce error, in this case will compute neighbors and try again
            print(f"Error: {e}. Trying to compute neighbors first and re-run.")
            scv.pp.neighbors(adata_sub)
            scv.tl.velocity_graph(adata_sub)
        # Compute and store the velocity embeddings
        scv.tl.velocity_embedding(adata_sub, basis='umap')
        test_sub = adata_sub.copy()

        # Set the colors in test_sub
        test_sub.uns[f'{sub_key}_colors'] = [expert_color_dict[expert] for expert in expert_labels]

        # Plot velocity embedding stream for full data's corresponding part
        scv.pl.velocity_embedding_stream(
            test_full,
            basis='X_umap',
            color=sub_key,
            groups=part,
            ax=axes[0, i],
            show=False,
            min_mass=2,
            density=0.5,
            arrow_size=0.5,
            linewidth=0.5,
            xlabel='UMAP1',
            ylabel='UMAP2',
            vkey='velocity',
            title=f'Full Data\n{sub_key} {part}'
        )

        # Record the x and y limits
        xlims = axes[0, i].get_xlim()
        ylims = axes[0, i].get_ylim()

        # Plot velocity embedding stream for sub data
        scv.pl.velocity_embedding_stream(
            test_sub,
            basis='X_umap',
            color=sub_key,
            ax=axes[1, i],
            show=False,
            min_mass=2,
            density=0.5,
            arrow_size=0.5,
            linewidth=0.5,
            xlabel='UMAP1',
            ylabel='UMAP2',
            vkey='velocity',
            title=f'Subset Data\n{sub_key} {part}'
        )

        # Set the x and y limits to match the full data's plot
        axes[1, i].set_xlim(xlims)
        axes[1, i].set_ylim(ylims)

        # Remove y-axis labels for inner plots to reduce clutter
        if i > 0:
            axes[0, i].set_ylabel('')
            axes[1, i].set_ylabel('')

        # Align the cells by their unique identifiers
        # Get the intersection of cell names
        common_cells = test_full_part.obs_names.intersection(test_sub.obs_names)

        # Ensure the cell ordering is the same in both datasets
        velocity_umap_full_aligned = pd.DataFrame(
            velocity_umap_full,
            index=test_full_part.obs_names
        ).loc[common_cells].values

        velocity_umap_sub_aligned = pd.DataFrame(
            test_sub.obsm['velocity_umap'],
            index=test_sub.obs_names
        ).loc[common_cells].values

        # Compute cosine similarity between aligned cells
        velocity_umap_similarity = cosine_similarity(
            velocity_umap_full_aligned, velocity_umap_sub_aligned
        ).diagonal()

        # Collect similarities and labels
        all_similarities.extend(velocity_umap_similarity)
        all_labels.extend([part] * len(velocity_umap_similarity))

    # Adjust layout and save the combined figure
    plt.tight_layout()
    if show:
        plt.savefig(f'{result_path}velocity_comparison_all_parts_{save_name}', bbox_inches='tight')
        plt.show()
    else:
        plt.show()
    plt.close(fig)

    # Create a DataFrame for plotting
    df_similarity = pd.DataFrame({
        'Cosine Similarity': all_similarities,
        sub_key: all_labels
    })

    # Create a palette for the violin plot
    palette = [expert_color_dict[part] for part in parts]

    # Plot violin plot for all parts
    plt.figure(figsize=size)
    sns.violinplot(
        x=sub_key,
        y='Cosine Similarity',
        data=df_similarity,
        inner='box',
        palette=palette
    )
    plt.title("Cosine Similarity for Velocity")
    plt.xlabel(sub_key)
    plt.ylabel("Cosine Similarity")
    plt.xticks(rotation=90)
    plt.tight_layout()
    if show:
        plt.savefig(f'{result_path}cosine_similarity_violin_{save_name}', bbox_inches='tight')
        plt.show()
    else:
        plt.show()
    plt.close()

def find_pass_markers(
    adata,
    cluster_key,
    deg_params=None,
    layer=None,
    show_clusters=None,
    cutoff=0.05
):
    """
    Finds marker genes for each cluster.

    Parameters:
    - adata: AnnData object.
    - cluster_key: str, key in adata.obs to group cells.
    - deg_params: dict, additional parameters for DEG (e.g., {'method': 'wilcoxon'}).
    - layer: str, layer in adata to use for DEG (default uses adata.X).
    - show_clusters: list, clusters to process (default processes all clusters).
    - cutoff: adj pvalues cutoff values

    Returns:
    - marker_genes: dict, marker genes for each cluster.
    """
    if deg_params is None:
        deg_params = {}
    if show_clusters is None:
        show_clusters = adata.obs[cluster_key].unique().tolist()
    
    # Perform DEG analysis
    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_key,
        use_raw=False,
        layer=layer,
        **deg_params
    )
    
    # Dictionary to store marker genes
    marker_genes = {}
    
    for cluster in show_clusters:
        # Get DEG results for the cluster
        deg_df = sc.get.rank_genes_groups_df(adata, group=cluster,pval_cutoff=cutoff)
        # Select top N genes
        #top_deg = deg_df.head(top_n_genes)
        # Get gene names
        #gene_list = top_deg['names'].tolist()
        gene_list = deg_df['names'].tolist()
        
        marker_genes[cluster] = gene_list
    
    return marker_genes
