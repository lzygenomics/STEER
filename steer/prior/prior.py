import pandas as pd
import numpy as np
from scipy.stats import entropy, mannwhitneyu, rankdata
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from collections import defaultdict
from scipy.interpolate import LSQUnivariateSpline
import anndata as ad
from sklearn.cluster import MiniBatchKMeans
from ..plot import plot_entropy


def use_scaled_orig(df, use_orig = False):
    df['scale_orig_unsplice'] =  df.groupby('gene_name')['orig_unsplice'].transform(lambda x: x / x.max())
    df['scale_orig_splice'] =  df.groupby('gene_name')['orig_splice'].transform(lambda x: x / x.max())
    if use_orig:
        df['unsplice'] = df['scale_orig_unsplice']
        df['splice'] = df['scale_orig_splice']
    return df

def assign_random_clusters(data, labels): # type: ignore
    data['random_cluster'] = data.groupby('gene_name')['gene_name'].transform(lambda x: np.random.choice(labels, size=len(x)))
    return data

def assign_random_clusters_to_cells(adata, labels):
    # Assign random clusters to cells
    random_clusters = np.random.choice(labels, size=adata.n_obs)
    adata.obs['random_cluster'] = random_clusters
    return adata

def add_annotations_to_df(df: pd.DataFrame, adata: ad.AnnData, annotation_cols: list) -> pd.DataFrame:

    annotations = adata.obs[annotation_cols].copy()
    df.set_index('cellID', drop=False, inplace=True)
    # Merge the DataFrame with the annotations DataFrame
    df = df.merge(annotations, left_index = True, right_index = True, how='left')

    return df


#### Part 1: Filter Noise/Time-irrelative genes based on Entropy ####

# Calculate Entropy
def grid_us_space_extended(data: pd.DataFrame, bins: int, label_cols: list, min_ncell: int) -> pd.DataFrame:

    # Define bins for 'unsplice' and 'splice'
    data['unsplice_bin'] = pd.cut(data['unsplice'], bins=bins, labels=False)
    data['splice_bin'] = pd.cut(data['splice'], bins=bins, labels=False)

    # Store the results
    results = []
    
    for gene, group in data.groupby('gene_name'):
        gene_results = {'gene_name': gene}
        for label_col in label_cols:
            # Calculate entropy for each grid for the current label
            entropies = group.groupby(['unsplice_bin', 'splice_bin', label_col]).size().unstack(fill_value=0)
            entropies = entropies[entropies.sum(axis=1) >= min_ncell]  # Filter grids with less than 5 cells

            # Calculate entropy for filtered grids
            grid_entropies = entropies.apply(lambda x: entropy(x, base=2), axis=1)
            mean_entropy = grid_entropies.mean() if not grid_entropies.empty else np.nan
            
            # Update results
            gene_results[f'mean_entropy_{label_col}'] = mean_entropy
        
        results.append(gene_results)
    result = pd.DataFrame(results)
    result.columns = ['gene_name'] + ['Entropy_' + col for col in label_cols]
    result = result.sort_values('Entropy_pred_cluster')
    # Return mean entropy for each gene and label type
    return result

def filter_by_entropy(pretrain_df, Entropy_df, keep_ngene,sort_by='Entropy_pred_cluster'):
    # Merge the dataframes on 'gene_name'
    merged_df = pd.merge(pretrain_df, Entropy_df, on='gene_name', how='left')

    # Identify the top 1000 unique genes with the lowest 'Entropy_Pred_Cluster'
    top_genes = Entropy_df.nsmallest(keep_ngene, sort_by)['gene_name']

    # Filter the merged dataframe to keep only rows corresponding to the top 1000 genes
    filtered_df = merged_df[merged_df['gene_name'].isin(top_genes)]
    return filtered_df

def add_entropy_to_adata(adata: ad.AnnData, entropy_df: pd.DataFrame, top_n: int = 1000,sort_by: str = 'Entropy_pred_cluster') -> ad.AnnData:
    # Ensure the gene_name in entropy_df is the index of adata.var
    entropy_df = entropy_df.set_index('gene_name')
    
    # Check if the gene names match between entropy_df and adata.var
    if not entropy_df.index.isin(adata.var_names).all():
        raise ValueError("Some gene names in entropy_df do not match the gene names in adata.var_names")
    
    # Add the columns to adata.var using the names from entropy_df
    for column in entropy_df.columns:
        adata.var[column] = entropy_df[column]

    # Sort by the specified column and select the top_n genes
    if sort_by not in entropy_df.columns:
        raise ValueError(f"The specified sort_by column '{sort_by}' is not found in entropy_df.")
    # Sort by Entropy_pred_cluster and select the top_n genes
    top_genes = entropy_df.sort_values(by=sort_by, ascending=True).head(top_n).index
    
    # Create a boolean column in adata.var to indicate velocity genes
    adata.var['is_velocity_gene'] = adata.var_names.isin(top_genes)
    
    return adata

#### Part 2: Label the Mono-gene by showing the marginal distribution of differernt patterns

def analyze_density_peaks_for_genes(data, bins, cutoff = 70, smoothed = False, neighbor_count=10):
    gene_results = {}

    unique_genes = data['gene_name'].unique()
    data['splice_bin'] = pd.cut(data['splice'], bins=bins, labels=range(bins))
    
    for gene in unique_genes:
        # print(gene)
        gene_data = data[data['gene_name'] == gene]

        bin_counts = {'One group': 0, 'Multiple groups': 0, 'No clear peak': 0, 'No data': 0}

        for bin_label in range(bins):
            bin_data = gene_data[gene_data['splice_bin'] == bin_label]

            if bin_data.shape[0] < 2:
                bin_counts['No data'] += 1
            else:
                unspliced_values = bin_data['unsplice'].values
                if smoothed == False:
                    if len(unspliced_values) >= neighbor_count:
                        smooth_unspliced_values = np.convolve(unspliced_values, np.ones(neighbor_count)/neighbor_count, mode='same')
                    else:
                        smooth_unspliced_values = np.convolve(unspliced_values, np.ones(len(unspliced_values))/len(unspliced_values), mode='same')

                    peaks, _ = find_peaks(smooth_unspliced_values, prominence=0.01, height=0.01)
                else:
                    smooth_unspliced_values = unspliced_values
                    peaks, _ = find_peaks(smooth_unspliced_values, prominence=0.01, height=0.01)
                if len(peaks) == 0:
                    bin_counts['No clear peak'] += 1
                else:
                    peak_clusters = bin_data.iloc[peaks]['pred_cluster'].values
                    unique_clusters = np.unique(peak_clusters)

                    if len(unique_clusters) == 1:
                        bin_counts['One group'] += 1
                    else:
                        bin_counts['Multiple groups'] += 1

        total_bins = sum(bin_counts.values())
        one_group_percentage = ((bin_counts['One group']+bin_counts['No clear peak']) / total_bins * 100) if total_bins > 0 else 0
        gene_results[gene] = one_group_percentage

    data['pred_gene_type'] = data['gene_name'].map(gene_results)

    data['pred_gene_type'] = data['pred_gene_type'].apply(lambda x: 'Mono_gene' if x >= cutoff else 'Compo_gene')

    return data

#### Part 3: Further Lable the genes and cells by using Clustering information
# Function to handle non-unique x values
def aggregate_y_by_x(x, y):
    xy_dict = {}
    for xi, yi in zip(x, y):
        xi = float(xi)
        yi = float(yi)
        if xi not in xy_dict:
            xy_dict[xi] = []
        xy_dict[xi].append(yi)
    x_unique = np.array(sorted(xy_dict.keys()))
    y_aggregated = np.array([np.mean(xy_dict[xi]) for xi in x_unique])
    return x_unique, y_aggregated

# Function to compute convexity using LSQ B-spline fitting
def compute_convexity(x, y, num_knots=5):
    # Aggregate y values for non-unique x values
    x_unique, y_aggregated = aggregate_y_by_x(x, y)

    # Ensure num_knots is appropriate given the number of data points and the order of the spline (k=2)
    k = 2
    max_knots = len(x_unique) - (k + 1)
    if num_knots > max_knots:
        num_knots = max_knots

    # Ensure there are enough unique x values for the number of knots
    if len(x_unique) <= k:
        raise ValueError("Not enough unique x values to fit the spline.")

    # Check for valid x and y values (no NaNs or Infs)
    if np.any(np.isnan(x_unique)) or np.any(np.isnan(y_aggregated)) or np.any(np.isinf(x_unique)) or np.any(np.isinf(y_aggregated)):
        raise ValueError("Input data contains NaNs or Infs.")

    # Ensure x_unique is sorted and does not contain duplicates
    if not np.all(np.diff(x_unique) > 0):
        raise ValueError("x_unique must be sorted and contain unique values.")

    # Fit a LSQ B-spline to the data
    try:
        # Calculate the internal knots, excluding the first and last point
        knots = np.linspace(x_unique[1], x_unique[-2], num_knots - 2)
        spline = LSQUnivariateSpline(x_unique, y_aggregated, t=knots, k=k)
    except Exception as e:
        print(f"x_unique: {x_unique}")
        print(f"y_aggregated: {y_aggregated}")
        print(f"knots: {knots}")
        raise ValueError(f"Error fitting LSQ B-spline: {e}")

    # Compute the second derivative of the spline
    second_derivative = spline.derivative(n=2)(x)
    
    # Calculate the 1% and 99% quantiles of the second derivative
    quantile_01 = np.quantile(second_derivative, 0.05)
    quantile_99 = np.quantile(second_derivative, 0.95)
    
    # Filter the second derivative to exclude values below the 1% quantile and above the 99% quantile
    filtered_second_derivative = second_derivative[(second_derivative >= quantile_01) & (second_derivative <= quantile_99)]
    ranks = rankdata(np.abs(filtered_second_derivative))
    # Calculate the number of elements greater than 0 and less than 0 in the filtered second derivative
    # Separate positive and negative ranks
    positive_ranks = ranks[filtered_second_derivative > 0]
    negative_ranks = ranks[filtered_second_derivative < 0]

    # sum_positive_ranks = positive_ranks.sum()
    # sum_negative_ranks = negative_ranks.sum()
    # diff = sum_positive_ranks - sum_negative_ranks
    
    # Perform the Wilcoxon signed-rank test
    if len(positive_ranks) and len(negative_ranks) > 0:
        _, p_value_greater = mannwhitneyu(positive_ranks, negative_ranks, alternative ='greater')
        _, p_value_less = mannwhitneyu(positive_ranks, negative_ranks, alternative ='less')
        diff = p_value_less - p_value_greater
        p_value = np.min([p_value_greater, p_value_less])
    elif len(positive_ranks) == 0:
        diff = -1
        p_value = 0  # Handle the case where there are no values after filtering
    elif len(negative_ranks) == 0:
        diff = 1
        p_value = 0
    # Return convexity measure (num_positive - num_negative), p-value, and second derivative
    return diff, p_value, second_derivative # num_positive - num_negative

def pred_regulation(datai):
    # Precompute and store unique genes and clusters
    data = datai#[(data['pred_group_type'] == 'Compo_gene')]
    # compo_genes_data['splice'] = compo_genes_data['recon_s']
    # compo_genes_data['unsplice'] = compo_genes_data['recon_u']
    grouped_genes = data.groupby('gene_name')
    
    for gene, gene_data in grouped_genes:
        print(gene)
        gene_indices = gene_data.index
        clusters = gene_data['pred_cluster'].unique()
        
        for cluster in clusters:
            cluster_data = gene_data[gene_data['pred_cluster'] == cluster]
            cell_indices = cluster_data.index
            # # Filter out zero unsplice and splice values
            # if filter_zero:
            #     cluster_data = cluster_data.loc[(cluster_data['unsplice'] != 0) & (cluster_data['splice'] != 0)]
            x_cluster = cluster_data['splice']
            y_cluster = cluster_data['unsplice']

            if len(np.unique(x_cluster)) < 10 or len(np.unique(y_cluster)) < 10:
                data.loc[cell_indices, 'pred_cell_type'] = 'Auto'
                continue  # Skip to the next cluster
            fs, ps, ds = compute_convexity(x_cluster, y_cluster)
            fu, pu, du = compute_convexity(y_cluster, x_cluster)
            data.loc[cell_indices, 'du'] = du
            data.loc[cell_indices, 'ds'] = ds
            data.loc[cell_indices, 'fu'] = fu
            data.loc[cell_indices, 'fs'] = fs
            data.loc[cell_indices, 'pu'] = pu
            data.loc[cell_indices, 'ps'] = ps

            if (fs > 0) & (fu < 0):
                data.loc[cell_indices, 'pred_cell_type'] = 'Down'
                data.loc[cell_indices, 'pred_group_type'] = 'Compo_Down'
                data.loc[cell_indices, 'confidence'] = min(pu, ps)
            elif (fs < 0) & (fu > 0):
                data.loc[cell_indices, 'pred_cell_type'] = 'Up'
                data.loc[cell_indices, 'pred_group_type'] = 'Compo_Up'
                data.loc[cell_indices, 'confidence'] = min(pu, ps)
            elif pu <= ps:
                if fu > 0:
                    data.loc[cell_indices, 'pred_cell_type'] = 'Up'
                    data.loc[cell_indices, 'pred_group_type'] = 'Compo_Up'
                    data.loc[cell_indices, 'confidence'] = pu
                else:
                    data.loc[cell_indices, 'pred_cell_type'] = 'Down'
                    data.loc[cell_indices, 'pred_group_type'] = 'Compo_Down'
                    data.loc[cell_indices, 'confidence'] = pu
            elif ps < pu:
                if fs > 0:
                    data.loc[cell_indices, 'pred_cell_type'] = 'Down'
                    data.loc[cell_indices, 'pred_group_type'] = 'Compo_Down'
                    data.loc[cell_indices, 'confidence'] = ps
                else:
                    data.loc[cell_indices, 'pred_cell_type'] = 'Up'
                    data.loc[cell_indices, 'pred_group_type'] = 'Compo_Up'
                    data.loc[cell_indices, 'confidence'] = ps
    return data

def pred_regulation_anndata(adata):
    # Precompute and store unique genes and clusters
    gene_names = adata.var_names
    clusters = adata.obs['pred_cluster'].unique()

    # Create new layers to store results
    adata.layers['pred_cell_type'] = np.zeros(adata.shape, dtype='U10')  # String type with max length 10
    adata.layers['pred_group_type'] = np.zeros(adata.shape, dtype='U10')
    adata.layers['confidence'] = np.zeros(adata.shape, dtype=float)
    adata.layers['du'] = np.zeros(adata.shape, dtype=float)
    adata.layers['ds'] = np.zeros(adata.shape, dtype=float)
    adata.layers['fu'] = np.zeros(adata.shape, dtype=float)
    adata.layers['fs'] = np.zeros(adata.shape, dtype=float)
    adata.layers['pu'] = np.zeros(adata.shape, dtype=float)
    adata.layers['ps'] = np.zeros(adata.shape, dtype=float)

    for gene in gene_names:
        print(gene)
        gene_idx = adata.var_names.get_loc(gene)  # Get the index of the gene
        
        for cluster in clusters:
            cluster_data = adata[adata.obs['pred_cluster'] == cluster, gene]
            cell_indices = np.where(adata.obs['pred_cluster'] == cluster)[0]
            
            x_cluster = cluster_data.layers['scale_Ms'].flatten()
            y_cluster = cluster_data.layers['scale_Mu'].flatten()

            if len(np.unique(x_cluster)) < 10 or len(np.unique(y_cluster)) < 10:
                adata.layers['pred_cell_type'][cell_indices, gene_idx] = 'Auto'
                continue  # Skip to the next cluster

            fs, ps, ds = compute_convexity(x_cluster, y_cluster)
            fu, pu, du = compute_convexity(y_cluster, x_cluster)

            adata.layers['du'][cell_indices, gene_idx] = du
            adata.layers['ds'][cell_indices, gene_idx] = ds
            adata.layers['fu'][cell_indices, gene_idx] = fu
            adata.layers['fs'][cell_indices, gene_idx] = fs
            adata.layers['pu'][cell_indices, gene_idx] = pu
            adata.layers['ps'][cell_indices, gene_idx] = ps

            if (fs > 0) & (fu < 0):
                adata.layers['pred_cell_type'][cell_indices, gene_idx] = 'Down'
                adata.layers['pred_group_type'][cell_indices, gene_idx] = 'Compo_Down'
                adata.layers['confidence'][cell_indices, gene_idx] = min(pu, ps)
            elif (fs < 0) & (fu > 0):
                adata.layers['pred_cell_type'][cell_indices, gene_idx] = 'Up'
                adata.layers['pred_group_type'][cell_indices, gene_idx] = 'Compo_Up'
                adata.layers['confidence'][cell_indices, gene_idx] = min(pu, ps)
            elif pu <= ps:
                if fu > 0:
                    adata.layers['pred_cell_type'][cell_indices, gene_idx] = 'Up'
                    adata.layers['pred_group_type'][cell_indices, gene_idx] = 'Compo_Up'
                    adata.layers['confidence'][cell_indices, gene_idx] = pu
                else:
                    adata.layers['pred_cell_type'][cell_indices, gene_idx] = 'Down'
                    adata.layers['pred_group_type'][cell_indices, gene_idx] = 'Compo_Down'
                    adata.layers['confidence'][cell_indices, gene_idx] = pu
            elif ps < pu:
                if fs > 0:
                    adata.layers['pred_cell_type'][cell_indices, gene_idx] = 'Down'
                    adata.layers['pred_group_type'][cell_indices, gene_idx] = 'Compo_Down'
                    adata.layers['confidence'][cell_indices, gene_idx] = ps
                else:
                    adata.layers['pred_cell_type'][cell_indices, gene_idx] = 'Up'
                    adata.layers['pred_group_type'][cell_indices, gene_idx] = 'Compo_Up'
                    adata.layers['confidence'][cell_indices, gene_idx] = ps

    return adata

class PriorInferenceManager:
    def __init__(self, adata, df, result_path, seed=618):
        self.adata = adata
        self.df = df
        self.result_path = result_path
        self.seed = seed
        
        # 备份原始 Expert Cluster (假设初始的 pred_cluster 就是 Expert)
        if 'expert_cluster' not in self.adata.obs:
            self.adata.obs['expert_cluster'] = self.adata.obs['pred_cluster'].copy()
            
        # 确保 df 的 cellID 是字符串，方便后续映射
        if 'cellID' in self.df.columns:
            self.df['cellID'] = self.df['cellID'].astype(str)

    def _update_mapping(self, cluster_col):
        """
        核心同步函数：确保 adata.obs['pred_cluster'] 和 df['pred_cluster'] 
        内容一致，且与指定的 cluster_col (Expert 或 Fine) 同步。
        """
        # 1. 更新 adata: 将目标列覆盖到 pred_cluster
        #    这是必须的，因为 steer.pred_regulation 默认读取 pred_cluster
        self.adata.obs['pred_cluster'] = self.adata.obs[cluster_col]
        
        # 2. 更新 df: 基于 Barcode String Mapping
        cluster_map = self.adata.obs[cluster_col].to_dict()
        self.df['pred_cluster'] = self.df['cellID'].map(cluster_map)
        
        # 3. 处理 NaN (Robust)
        if self.df['pred_cluster'].isna().any():
            mode_val = self.adata.obs[cluster_col].mode()[0]
            print(f"  [Info] Please Check ! Filling NaNs in df mapping with mode: {mode_val}")
            self.df['pred_cluster'] = self.df['pred_cluster'].fillna(mode_val)
            
        return self.adata, self.df

    def task1_define_fine_clusters(self, method='hierarchical', target_size=100):
        """
        任务 1: 定义 Fine Cluster
        :param method: 
            - 'global': 全局 K-Means
            - 'hierarchical': Expert 内部细分 (推荐)
            - 'none': 不进行细分，直接使用 Expert 作为 Fine Cluster (满足你的需求1)
        """
        print(f"--- Task 1: Generating Fine Clusters (Method: {method}) ---")
        
        X_emb = self.adata.obsm['X_pre_embed']
        
        if method == 'none':
            # 【新功能】直接复用 Expert
            print("  -> Method is 'none', copying Expert clusters to Fine clusters.")
            self.adata.obs['fine_cluster'] = self.adata.obs['expert_cluster'].astype(str)
            
        elif method == 'global':
            n_clusters = max(int(self.adata.n_obs / target_size), 5)
            print(f"  -> Global K-Means: Partitioning into {n_clusters} clusters.")
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=self.seed, batch_size=4096, n_init=3)
            labels = kmeans.fit_predict(X_emb)
            self.adata.obs['fine_cluster'] = labels.astype(str)
            
        elif method == 'hierarchical':
            print(f"  -> Hierarchical K-Means: Target size {target_size}")
            self.adata.obs['fine_cluster'] = self.adata.obs['expert_cluster'].astype(str)
            unique_experts = self.adata.obs['expert_cluster'].unique()
            
            count = 0
            for expert in unique_experts:
                idx = np.where(self.adata.obs['expert_cluster'] == expert)[0]
                n_cells = len(idx)
                
                if n_cells > 1.5 * target_size:
                    k_sub = max(int(n_cells / target_size), 2)
                    kmeans = MiniBatchKMeans(n_clusters=k_sub, random_state=self.seed, batch_size=4096, n_init=3)
                    sub_labels = kmeans.fit_predict(X_emb[idx])
                    new_labels = [f"{expert}_{l}" for l in sub_labels]
                    
                    col_idx = self.adata.obs.columns.get_loc('fine_cluster')
                    self.adata.obs.iloc[idx, col_idx] = new_labels
                    count += k_sub
                else:
                    count += 1
            print(f"  -> Generated {count} hierarchical micro-clusters.")
        
        return self.adata

    def task2_filter_genes(self, based_on='expert', keep_ngene=1000, use_filter=True):
        """
        任务 2: 基因筛选
        """
        target_col = 'expert_cluster' if based_on == 'expert' else 'fine_cluster'
        # 同步状态
        self.adata, self.df = self._update_mapping(target_col)
        pretrain_df = use_scaled_orig(self.df, use_orig=False)
        # 这里的 bins 也可以根据是 expert 还是 fine 自动调整
        bins = 5
        
        if not use_filter:
            print(f"--- Task 2: Gene Filtering Skipped (Calculated on {target_col} for reference) ---")
            Entropy_df = grid_us_space_extended(pretrain_df, bins=bins, label_cols=['pred_cluster'], min_ncell=5)
            self.adata = add_entropy_to_adata(self.adata, Entropy_df, top_n=self.adata.n_vars)
        else:
            print(f"--- Task 2: Filtering Genes (Based on: {based_on.upper()}, Keep: {keep_ngene}) ---")
            Entropy_df = grid_us_space_extended(pretrain_df, bins=bins, label_cols=['pred_cluster'], min_ncell=5)
            # 注意 key 是 pred_cluster
            plot_entropy(Entropy_df, path=self.result_path, key='Entropy_pred_cluster')
            self.adata = add_entropy_to_adata(self.adata, Entropy_df, top_n=keep_ngene)
            
        return self.adata

    def task3_calc_convexity(self, based_on='expert'):
        """
        任务 3: 计算凸性 (Prior Direction)
        """
        print(f"--- Task 3: Calculating Direction (Based on: {based_on.upper()}) ---")
        target_col = 'expert_cluster' if based_on == 'expert' else 'fine_cluster'
        # 关键：切换 mapping，让 pred_regulation 看到正确的列
        self.adata, self.df = self._update_mapping(target_col)
        # 计算
        self.adata = pred_regulation_anndata(self.adata)
        return self.adata
    
    def finalize_for_training(self):
        """
        收尾与任务 4 准备: 
        1. 恢复 Expert 标签用于展示
        2. 返回处理好的 fine_cluster_vector 给 PyG
        """
        print("--- Finalizing: Restoring Expert labels & Preparing Fine Cluster Vector ---")
        
        # 1. 恢复展示用的 Cluster
        self.adata.obs['pred_cluster'] = self.adata.obs['expert_cluster']
        
        # 2. 准备 Task 4 所需的向量 (Correlation Cluster)
        # 逻辑：如果有 fine_cluster 就用，没有就回退到 expert
        if 'fine_cluster' in self.adata.obs:
            cluster_series = self.adata.obs['fine_cluster'].astype(str)
        else:
            cluster_series = self.adata.obs['pred_cluster'].astype(str)
            
        # 转为整数编码
        fine_clus_vec_np = cluster_series.astype('category').cat.codes.values.astype(int)
        
        return self.adata, self.df, fine_clus_vec_np