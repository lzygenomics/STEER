import scanpy as sc
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from pyecharts.charts import Sankey
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from scipy.ndimage import gaussian_filter
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
# scvelo's velocity ploting import
import os
import warnings
import numpy as np
from scipy.sparse import coo_matrix, issparse

from scvelo import logging as logg
from scvelo import settings
from scvelo.core import get_n_jobs, l2_norm, parallelize
from scvelo.preprocessing.moments import get_moments
from scvelo.preprocessing.neighbors import (
    get_n_neighs,
    get_neighs,
    neighbors,
    pca,
    verify_neighbors,
)

from scvelo.tools import velocity
from scvelo.core import sum as sc_sum
import matplotlib.patches as mpatches
from collections import defaultdict

def embedding_plot(embeddings_df, fig_path, color_col = 'predicted_cluster'):

    # Step 2: Create an AnnData object with embeddings and cluster information
    adata = sc.AnnData(X=embeddings_df.iloc[:, :embeddings_df.shape[1] - 3])
    adata.obs = embeddings_df.loc[:, [color_col]]  # Add cluster information as observation annotation

    # Step 3: Run UMAP and plot
    sc.settings.figdir = fig_path
    sc.pp.neighbors(adata, use_rep='X')  # Compute the neighborhood graph
    sc.tl.umap(adata,min_dist=0.2)  # Compute UMAP
    sc.pl.umap(adata, color=color_col, title='UMAP projection colored by predicted cluster', save='Embedding_by_pred_cluster_refine.pdf')

    return('Plot Complete!')

# Updated function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    # Avoid division errors or invalid operations
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return np.nan
    return 1 - cosine(vec1, vec2)

# Updated function to process each group and keep track of identifiers
def process_group(group, gene_pattern):
    # Keep identifiers
    identifiers = group[['cellID', 'gene_name']]
    # Extract clusters for each entry
    clusters = group['clusters'].values
    # Vectorized computation of cosine similarity
    pred_vectors = group[['pred_vu', 'pred_vs']].values
    true_vectors = group[['ture_vu', 'ture_vs']].values
    # Calculate cosine similarity for each row in the group
    cosine_similarities = np.array([cosine_similarity(pred, true) for pred, true in zip(pred_vectors, true_vectors)])
    return pd.DataFrame({
        'cellID': identifiers['cellID'],
        'gene_name': identifiers['gene_name'],
        'Cosine_Similarity': cosine_similarities,
        'Cluster': clusters,
        'Gene_pattern': np.repeat(gene_pattern, len(cosine_similarities))
    })

def Cluster_Results_plot(final_df, fig_path):
    # Group the DataFrame by 'Gene_pattern' and process
    grouped = final_df.groupby(['Gene_pattern'])

    # Using Parallel processing to calculate results
    results_dfs = Parallel(n_jobs=6)(delayed(process_group)(group, name) for name, group in grouped)

    # Concatenate all results into a single DataFrame
    results_df = pd.concat(results_dfs, ignore_index=True)

    # Merge results back into the original DataFrame
    final_df_with_cosine = final_df.merge(results_df, on=['cellID', 'gene_name'], how='left')

    # Plotting
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Gene_pattern_x', y='Cosine_Similarity', hue='pred_cluster_refine', data=final_df_with_cosine, palette='Set2')
    plt.xticks(rotation=45)
    plt.title('Cosine Similarity by Gene Pattern and Cluster')
    plt.ylabel('Cosine Similarity')
    plt.xlabel('Gene Pattern')
    plt.legend(title='Cluster', loc='upper right')
    plt.tight_layout()
    plt.savefig(fig_path + 'Cosine_for_predicted_clusters.pdf')
    return('Plot Complete!')

def plot_time_points_from_dataframe(df, time_column, fig_path):
    """
    Plot the time points from a dataframe column to check for monotonicity.

    Parameters:
    df (pd.DataFrame): DataFrame containing the time points
    time_column (str): Name of the column containing the time points
    """
    # Extract time points from the dataframe
    time_points = df[time_column].values
    
    # Plot the time points against the row sequence
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(time_points)), time_points, s=10, alpha=0.6)
    plt.xlabel("Row Index")
    plt.ylabel("Time Points")
    plt.title("Time Points vs. Row Index")
    plt.savefig(fig_path + 'Time.pdf')

def plot_predTime_vs_trueTime(embeddings_df, fig_path):
    # Adding cluster information
    embeddings_df['cluster'] = np.where(embeddings_df.index < 1000, 'Cluster 1', 'Cluster 2')
    # Normalizing the time within each cluster
    embeddings_df['normalized_time'] = embeddings_df.groupby('cluster')['time'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Creating separate dataframes for each cluster
    cluster_1_df = embeddings_df[embeddings_df['cluster'] == 'Cluster 1']
    cluster_2_df = embeddings_df[embeddings_df['cluster'] == 'Cluster 2']

    # Creating pivot tables for the heatmaps
    heatmap_data_cluster_1 = cluster_1_df.pivot_table(index=cluster_1_df.index, columns='cluster', values='normalized_time')
    heatmap_data_cluster_2 = cluster_2_df.pivot_table(index=cluster_2_df.index, columns='cluster', values='normalized_time')

    xticks_cluster_1 = np.linspace(0, 999, 5, dtype=int)
    xticks_cluster_2 = np.linspace(0, 999, 5, dtype=int)

    # Plotting the heatmaps in subplots
    fig, axes = plt.subplots(2, 1, figsize=(9, 6))

    sns.heatmap(heatmap_data_cluster_1.T, cmap="viridis", cbar=True, ax=axes[0])
    axes[0].set_title('Predicted Time for each cell in Cluster 1')
    axes[0].set_xlabel('Cell Generation Time')
    axes[0].set_ylabel('')
    axes[0].set_xticks(xticks_cluster_1)
    axes[0].set_xticklabels(xticks_cluster_1)

    sns.heatmap(heatmap_data_cluster_2.T, cmap="viridis", cbar=True, ax=axes[1])
    axes[1].set_title('Predicted Time for each cell in Cluster 2')
    axes[1].set_xlabel('Cell Generation Time')
    axes[1].set_ylabel('')
    axes[1].set_xticks(xticks_cluster_2)
    axes[1].set_xticklabels(xticks_cluster_2)

    plt.tight_layout()
    plt.savefig(fig_path + 'Time_heatmap.pdf')

def Time_plot(embeddings_df, fig_path):
    # Assuming z and t are the latent embeddings and predicted time points from your model
    plot_time_points_from_dataframe(embeddings_df, 'time', fig_path)
    plot_predTime_vs_trueTime(embeddings_df, fig_path)

def Time_plot_cluster(df, cluster_column, save_path):
    """
    Generate scatter plots for each cluster in the DataFrame and save the figure.

    Parameters:
    df (pd.DataFrame): DataFrame containing cell information with columns 'cellID', 'time', 'pred_time', and the specified cluster column.
    cluster_column (str): Column name in df that stores the cluster information.
    save_path (str): Path to save the generated figure.
    """
    df = df.drop_duplicates(subset='cellID')

    # Get the unique clusters
    clusters = df[cluster_column].unique()

    # Define the number of rows and columns for subplots
    n_clusters = len(clusters)
    n_cols = n_clusters
    n_rows = (n_clusters + n_cols - 1) // n_cols

    # Set figure size to maintain equal width and height for subplots
    subplot_size = 5  # Size of each subplot (5x5 inches)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * subplot_size, n_rows * subplot_size))
    # If only one cluster, 'axes' is not an array, make it a list
    if n_clusters == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Plot each cluster
    for idx, cluster in enumerate(clusters):
        cluster_data = df[df[cluster_column] == cluster]
        ax = axes[idx]
        ax.scatter(cluster_data['time'], cluster_data['pred_time'], alpha=0.6)
        ax.set_title(f'Cluster {cluster}')
        ax.set_xlabel('Real Time')
        ax.set_ylabel('Predicted Time')
        ax.grid(True)

    # Hide any empty subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(save_path + cluster_column + '_Time.pdf')
    plt.close()

def plot_regulation(final_df, regulation_df, genes, fig_path, labels = ['Partial_Up', 'Trans_Boost', 'Multi_Forward',  'Partial_Down', 'Multi_Back', 'Whole_Single']):
    # Adjust column names in regulation_df to match the gene_name format in final_df
    regulation_df.columns = genes
    # Reset index of regulation_df to merge on cellID
    regulation_df.reset_index(inplace=True)
    regulation_df.rename(columns={'index': 'cellID'}, inplace=True)
    # Melt the regulation_df to long format
    regulation_long_df = pd.melt(regulation_df, id_vars=['cellID'], var_name='gene_name', value_name='value')

    # Transform values to 'Up' or 'Down'
    regulation_long_df['regulation'] = regulation_long_df['value'].apply(lambda x: 'Up' if x > 0 else 'Down' if x < 0 else 'Neutral')

    # Drop the original value column as it's no longer needed
    regulation_long_df.drop(columns=['value'], inplace=True)
    # Merge the dataframes on cellID and gene_name
    final_regu = pd.merge(final_df, regulation_long_df, on=['cellID', 'gene_name'], how='left')

    cell_labels = ['Up','Down']
    # Create a confusion matrix
    confusion_matrix = pd.crosstab(final_regu['Gene_pattern'], final_regu['regulation'], rownames=['True'], colnames=['Predicted'], normalize=False)

    # Reindex the confusion matrix to ensure the order of rows and columns
    confusion_matrix = confusion_matrix.reindex(index=labels, columns=cell_labels, fill_value=0)

    confusion_matrix = (confusion_matrix.iloc[0:6,[0,1]]/250).round().astype(int)

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(confusion_matrix, annot=True,fmt="d", cmap='cividis', vmin=0, vmax=2000)

    # Get the colorbar object
    cbar = ax.collections[0].colorbar

    # Set the colorbar ticks to include 0 and 250
    cbar.set_ticks([0, 2000])
    cbar.set_ticklabels([f'0', f'2000'])
    plt.title('Confusion Heatmap of Predicted cell regulation(Initial)')
    plt.savefig(fig_path + 'Confusion_Heatmap_cell_refine.pdf')

def heatmap_confusion_gene(data, labels, gene_labels, savepath, label_column = 'Gene_pattern', pred_column = 'pred_gene_type', scale = 2000):
    # Define the order of the labels
    labels = labels
    gene_labels = gene_labels
    # Create a confusion matrix
    confusion_matrix = pd.crosstab(data[label_column], data[pred_column], rownames=['True'], colnames=['Predicted'], normalize=False)

    # Reindex the confusion matrix to ensure the order of rows and columns
    confusion_matrix = confusion_matrix.reindex(index=labels, columns=gene_labels, fill_value=0)

    confusion_matrix = (confusion_matrix.iloc[0:6,[0,1]]//scale)

    # Plotting the heatmap
    plt.figure(figsize=(6, 8))
    ax = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap='cividis', vmin=0, vmax=250)

    # Get the colorbar object
    cbar = ax.collections[0].colorbar

    # Set the colorbar ticks to include 0 and 250
    cbar.set_ticks([0, 250])
    cbar.set_ticklabels([f'0', f'250'])

    # Set the title
    plt.title('Confusion heatmap of Predicted gene patterns')
    plt.savefig(savepath)

def heatmap_confusion_cell(data, labels, cell_labels, savepath, label_column = 'Gene_pattern', pred_column = 'pred_cell_type',scale = 250):
    labels = labels
    cell_labels = cell_labels
    # Create a confusion matrix
    confusion_matrix = pd.crosstab(data[label_column], data[pred_column], rownames=['True'], colnames=['Predicted'], normalize=False)

    # Reindex the confusion matrix to ensure the order of rows and columns
    confusion_matrix = confusion_matrix.reindex(index=labels, columns=cell_labels, fill_value=0)

    confusion_matrix = (confusion_matrix.iloc[0:6,[0,1]]/scale).round().astype(int)

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(confusion_matrix, annot=True,fmt="d", cmap='cividis', vmin=0, vmax=2000)

    # Get the colorbar object
    cbar = ax.collections[0].colorbar

    # Set the colorbar ticks to include 0 and 250
    cbar.set_ticks([0, 2000])
    cbar.set_ticklabels([f'0', f'2000'])
    plt.title('Confusion Heatmap of Predicted cell regulation(Initial)')
    plt.savefig(savepath)

# mv use this old plot code
# def plot_gene_expression(
#     adata,
#     genes,
#     color_by,
#     path,
#     ncols,
#     velocity=True,
#     agg='mean',
#     x_key='splice',
#     y_key='unsplice',
#     pred_vs_key='pred_vs',
#     pred_vu_key='pred_vu',
#     point_size=40,
#     quiver_scale=20,
#     quiver_width=0.005,
#     grid_size=25,
#     sigma=0.1,
#     lowerb=0,
#     upperb=98,
#     xlab = 'Spliced',
#     ylab = 'Unspliced',
#     random_seed=None
# ):
#     """
#     Plot gene expression data with optional velocity arrows using grid-based sampling and density filtering.

#     Parameters:
#     -----------
#     adata : AnnData
#         The annotated data matrix.
#     genes : list of str
#         List of gene names to plot.
#     color_by : str
#         The observation (cell) annotation to color by.
#     path : str
#         File path to save the generated plot.
#     ncols : int
#         Number of columns in the subplot grid.
#     velocity : bool, optional
#         Whether to plot velocity arrows. Default is True.
#     agg : str, optional
#         Aggregation method for grid sampling ('mean' or other). Default is 'mean'.
#     x_key : str, optional
#         Layer key for spliced counts. Default is 'splice'.
#     y_key : str, optional
#         Layer key for unspliced counts. Default is 'unsplice'.
#     pred_vs_key : str, optional
#         Layer key for predicted spliced velocity. Default is 'pred_vs'.
#     pred_vu_key : str, optional
#         Layer key for predicted unspliced velocity. Default is 'pred_vu'.
#     point_size : int, optional
#         Size of scatter plot points. Default is 40.
#     quiver_scale : float, optional
#         Scale factor for quiver arrows. Default is 20.
#     quiver_width : float, optional
#         Width of quiver arrows. Default is 0.005.
#     grid_size : int, optional
#         Number of grid cells along each axis for sampling. Default is 25.
#     sigma : float, optional
#         Standard deviation for Gaussian smoothing of density. Default is 0.1.
#     lowerb : float, optional
#         Lower percentile threshold for density filtering. Default is 0.
#     upperb : float, optional
#         Upper percentile threshold for density filtering. Default is 98.
#     random_seed : int, optional
#         Seed for random number generator for reproducibility. Default is None.
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import matplotlib.patches as mpatches
#     from scipy.ndimage import gaussian_filter
#     from collections import defaultdict

#     # Optional: Set random seed for reproducibility
#     if random_seed is not None:
#         np.random.seed(random_seed)

#     # Determine the layout for the subplots
#     n_genes = len(genes)
#     if ncols is None:
#         ncols = int(np.ceil(np.sqrt(n_genes)))
#     nrows = int(np.ceil(n_genes / ncols))
#     present_genes = [gene for gene in genes if gene in adata.var_names]
#     absent_genes = [gene for gene in genes if gene not in adata.var_names]
#     # Filter the anndata object for the selected genes
#     gene_indices = [adata.var_names.get_loc(gene) for gene in present_genes]
#     if absent_genes:
#         print(f"The following genes are not found in 'var_names': {', '.join(absent_genes)}")

#     splice = adata.layers[x_key][:, gene_indices]
#     unsplice = adata.layers[y_key][:, gene_indices]
#     pred_vs = adata.layers[pred_vs_key][:, gene_indices]
#     pred_vu = adata.layers[pred_vu_key][:, gene_indices]

#     # Extract colors from the obs column
#     color_categories = adata.obs[color_by].astype('category')
#     color_codes = color_categories.cat.codes

#     # Check if a colormap is available in adata.uns
#     cmap_key = f"{color_by}_colors"
#     if cmap_key in adata.uns:
#         color_map = adata.uns[cmap_key]
#         colors = [color_map[i] for i in color_codes]
#         legend_elements = [
#             mpatches.Patch(color=color_map[i], label=category)
#             for i, category in enumerate(color_categories.cat.categories)
#         ]
#     else:
#         colors = color_codes
#         plt_cmap = 'viridis'
#         unique_codes = np.unique(color_codes)
#         legend_elements = [
#             mpatches.Patch(
#                 color=plt.get_cmap(plt_cmap)(code / len(unique_codes)),
#                 label=category
#             )
#             for code, category in zip(unique_codes, color_categories.cat.categories)
#         ]

#     # Create figure and axes for subplots
#     fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, 3 * nrows))
#     axs = axs.flatten() if n_genes > 1 else [axs]

#     for i, gene in enumerate(genes):
#         if gene not in adata.var_names:
#             # Skip genes not present
#             continue

#         # Scatter plot of unspliced vs spliced counts with color
#         if cmap_key in adata.uns:
#             axs[i].scatter(
#                 splice[:, i],
#                 unsplice[:, i],
#                 c=colors,
#                 alpha=0.5,
#                 edgecolor='none',
#                 s=point_size
#             )
#         else:
#             axs[i].scatter(
#                 splice[:, i],
#                 unsplice[:, i],
#                 c=color_codes,
#                 cmap=plt_cmap,
#                 alpha=0.5,
#                 edgecolor='none',
#                 s=point_size
#             )

#         # Initialize variables for velocity plotting
#         selected_splice = selected_unsplice = selected_vs = selected_vu = None

#         if velocity:
#             # --- Grid-Based Sampling ---

#             # Define the grid over the data range
#             x_min, x_max = splice[:, i].min(), splice[:, i].max()
#             y_min, y_max = unsplice[:, i].min(), unsplice[:, i].max()

#             x_bins = np.linspace(x_min, x_max, grid_size + 1)
#             y_bins = np.linspace(y_min, y_max, grid_size + 1)

#             # Digitize the data to assign each point to a grid cell
#             x_digitized = np.digitize(splice[:, i], bins=x_bins) - 1  # bins start from index 0
#             y_digitized = np.digitize(unsplice[:, i], bins=y_bins) - 1

#             # Ensure indices are within valid range
#             x_digitized = np.clip(x_digitized, 0, grid_size - 1)
#             y_digitized = np.clip(y_digitized, 0, grid_size - 1)

#             # Compute the 2D histogram counts
#             counts, _, _ = np.histogram2d(
#                 splice[:, i], unsplice[:, i], bins=[x_bins, y_bins]
#             )

#             # Smooth the counts with a Gaussian filter
#             counts_smoothed = gaussian_filter(counts, sigma=sigma)

#             # Flatten the smoothed counts to get density estimates
#             density_estimate = counts_smoothed.flatten()

#             # Compute the density threshold
#             lower_density_threshold = np.percentile(density_estimate, lowerb)
#             upper_density_threshold = np.percentile(density_estimate, upperb)
#             # Create a boolean mask of grid cells to keep
#             bool_density = (density_estimate > lower_density_threshold) & (density_estimate < upper_density_threshold)

#             # Reshape bool_density back to 2D grid
#             bool_density_2d = bool_density.reshape(counts_smoothed.shape)

#             # Map grid cell indices to data point indices
#             grid_cell_indices = x_digitized + y_digitized * grid_size

#             cell_to_indices = defaultdict(list)
#             for idx, grid_idx in enumerate(grid_cell_indices):
#                 cell_to_indices[grid_idx].append(idx)

#             # Get grid cell indices that pass the density threshold
#             passing_cells = np.argwhere(bool_density_2d)
#             passing_x_indices = passing_cells[:, 0]
#             passing_y_indices = passing_cells[:, 1]
#             grid_cell_indices_passing = passing_x_indices + passing_y_indices * grid_size

#             # Initialize lists to store selected data
#             selected_splice = []
#             selected_unsplice = []
#             selected_vs = []
#             selected_vu = []
#             # For each passing grid cell, compute mean positions and velocities
#             if agg == 'mean':
#                 for grid_idx in grid_cell_indices_passing:
#                     if grid_idx in cell_to_indices:
#                         indices_in_cell = cell_to_indices[grid_idx]
#                         splice_mean = splice[indices_in_cell, i].mean()
#                         unsplice_mean = unsplice[indices_in_cell, i].mean()
#                         vs_mean = pred_vs[indices_in_cell, i].mean()
#                         vu_mean = pred_vu[indices_in_cell, i].mean()

#                         selected_splice.append(splice_mean)
#                         selected_unsplice.append(unsplice_mean)
#                         selected_vs.append(vs_mean)
#                         selected_vu.append(vu_mean)
#             else:
#                 for grid_idx in grid_cell_indices_passing:
#                     if grid_idx in cell_to_indices:
#                         indices_in_cell = cell_to_indices[grid_idx]
#                         if len(indices_in_cell) == 0:
#                             continue  # Skip empty cells

#                         # Randomly select one index from the cell
#                         selected_idx = np.random.choice(indices_in_cell)
                        
#                         # Append the selected data point's values
#                         selected_splice.append(splice[selected_idx, i])
#                         selected_unsplice.append(unsplice[selected_idx, i])
#                         selected_vs.append(pred_vs[selected_idx, i])
#                         selected_vu.append(pred_vu[selected_idx, i])

#             # Handle case where no grid cells meet the density threshold
#             if len(selected_splice) == 0:
#                 print(f"No grid cells with density above the percentile for gene {gene}. Skipping quiver plot.")
#             else:
#                 # Convert lists to arrays
#                 selected_splice = np.array(selected_splice)
#                 selected_unsplice = np.array(selected_unsplice)
#                 selected_vs = np.array(selected_vs)
#                 selected_vu = np.array(selected_vu)

#         # Plot velocity arrows if velocity=True and selected data is available
#         if velocity and selected_splice is not None and len(selected_splice) > 0:
#             # Determine arrow colors based on vs and vu values
#             arrow_colors = np.where(
#                 (selected_vs > 0) & (selected_vu > 0),
#                 'black',
#                 'black'
#             )

#             # Plot velocity arrows using the selected data
#             axs[i].quiver(
#                 selected_splice,
#                 selected_unsplice,
#                 selected_vs,
#                 selected_vu,
#                 units='xy',
#                 angles='xy',
#                 scale=quiver_scale,
#                 width=quiver_width,
#                 color=arrow_colors
#             )
#         elif velocity:
#             print(f"Velocity arrows not plotted for gene {gene} due to insufficient data.")

#         # # Set the same limits for both x and y axes
#         # axs[i].set_xlim(0, 1)
#         # axs[i].set_ylim(0, 1)
#         # Label axes and set titles
#         axs[i].set_aspect('auto', adjustable='box')
#         axs[i].set_xlabel(xlab)
#         axs[i].set_ylabel(ylab)
#         axs[i].set_title(f"{gene}")

#     # Adjust layout to prevent subplots from being distorted
#     plt.subplots_adjust(wspace=0.4, hspace=0.4)

#     # Add a single legend to the figure to show the mapping of colors to categories
#     fig.legend(
#         handles=legend_elements,
#         title=color_by,
#         loc='upper center',
#         bbox_to_anchor=(0.5, 0.05),
#         ncol=len(legend_elements),
#         frameon=False
#     )

#     # Hide any unused subplots
#     for j in range(i + 1, len(axs)):
#         fig.delaxes(axs[j])

#     #plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # Adjust layout to make room for the legend
#     plt.savefig(path)
#     plt.close()

def sample_and_plot_us(data, savepath, x_name = 'scale_orig_splice', y_name = 'scale_orig_unsplice',color_name='pred_cluster_refine'):
    # Get unique Gene_patterns and determine the number of subplots needed (up to 6)
    patterns = data['Gene_pattern'].unique()
    num_plots = min(len(patterns), 6)
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    axes = axes.flatten()

    for i, pattern in enumerate(patterns[:num_plots]):
        # Filter data for the current pattern
        pattern_data = data[data['Gene_pattern'] == pattern]
        # Randomly select one gene
        selected_gene = np.random.choice(pattern_data['gene_name'].unique())
        gene_data = pattern_data[pattern_data['gene_name'] == selected_gene]

        # Scatter plot for the selected gene
        ax = axes[i]
        clusters = gene_data[color_name].unique()
        for cluster in clusters:
            cluster_data = gene_data[gene_data[color_name] == cluster]
            ax.scatter(cluster_data[x_name], cluster_data[y_name], s=1, label=f'Cluster {cluster}')

        #ax.scatter(gene_data[x_name], gene_data[y_name], c=gene_data['pred_cluster_refine'], s=5)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title(f'Gene Pattern: {pattern}')
        ax.set_aspect('equal', 'box')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
    # Turn off any unused axes
    for j in range(num_plots, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(savepath)


# TODO: Add docstrings
def get_indices(dist, n_neighbors=None, mode_neighbors="distances"):
    """TODO."""
    from scvelo.preprocessing.neighbors import compute_connectivities_umap

    D = dist.copy()
    D.data += 1e-6

    n_counts = sc_sum(D > 0, axis=1)
    n_neighbors = (
        n_counts.min() if n_neighbors is None else min(n_counts.min(), n_neighbors)
    )
    rows = np.where(n_counts > n_neighbors)[0]
    cumsum_neighs = np.insert(n_counts.cumsum(), 0, 0)
    dat = D.data

    for row in rows:
        n0, n1 = cumsum_neighs[row], cumsum_neighs[row + 1]
        rm_idx = n0 + dat[n0:n1].argsort()[n_neighbors:]
        dat[rm_idx] = 0
    D.eliminate_zeros()

    D.data -= 1e-6
    if mode_neighbors == "distances":
        indices = D.indices.reshape((-1, n_neighbors))
    elif mode_neighbors == "connectivities":
        knn_indices = D.indices.reshape((-1, n_neighbors))
        knn_distances = D.data.reshape((-1, n_neighbors))
        _, conn = compute_connectivities_umap(
            knn_indices, knn_distances, D.shape[0], n_neighbors
        )
        indices = get_indices_from_csr(conn)
    return indices, D


# TODO: Add docstrings
def get_indices_from_csr(conn):
    """TODO."""
    # extracts indices from connectivity matrix, pads with nans
    ixs = np.ones((conn.shape[0], np.max((conn > 0).sum(1)))) * np.nan
    for i in range(ixs.shape[0]):
        cell_indices = conn[i, :].indices
        ixs[i, : len(cell_indices)] = cell_indices
    return ixs


# TODO: Add docstrings
def get_iterative_indices(
    indices,
    index,
    n_recurse_neighbors=2,
    max_neighs=None,
):
    """TODO."""

    def iterate_indices(indices, index, n_recurse_neighbors):
        if n_recurse_neighbors > 1:
            index = iterate_indices(indices, index, n_recurse_neighbors - 1)
        ix = np.append(index, indices[index])  # direct and indirect neighbors
        if np.isnan(ix).any():
            ix = ix[~np.isnan(ix)]
        return ix.astype(int)

    indices = np.unique(iterate_indices(indices, index, n_recurse_neighbors))
    if max_neighs is not None and len(indices) > max_neighs:
        indices = np.random.choice(indices, max_neighs, replace=False)
    return indices

# TODO: Add docstrings
def vals_to_csr(vals, rows, cols, shape, split_negative=False):
    """TODO."""
    graph = coo_matrix((vals, (rows, cols)), shape=shape)

    if split_negative:
        graph_neg = graph.copy()

        graph.data = np.clip(graph.data, 0, 1)
        graph_neg.data = np.clip(graph_neg.data, -1, 0)

        graph.eliminate_zeros()
        graph_neg.eliminate_zeros()

        return graph.tocsr(), graph_neg.tocsr()

    else:
        return graph.tocsr()

def cosine_correlation(dX, Vi):
    """TODO."""
    dx = dX - dX.mean(-1)[:, None]
    Vi_norm = l2_norm(Vi, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if Vi_norm == 0:
            result = np.zeros(dx.shape[0])
        else:
            result = (
                np.einsum("ij, j", dx, Vi) / (l2_norm(dx, axis=1) * Vi_norm)[None, :]
            )
    return result

# TODO: Add docstrings
class VelocityGraph:
    """TODO."""

    def __init__(
        self,
        adata,
        vkey="velocity",
        xkey="Ms",
        tkey=None,
        basis=None,
        n_neighbors=None,
        sqrt_transform=None,
        n_recurse_neighbors=None,
        random_neighbors_at_max=None,
        gene_subset=None,
        approx=None,
        report=False,
        compute_uncertainties=None,
        mode_neighbors="distances",
    ):
        subset = np.ones(adata.n_vars, bool)
        if gene_subset is not None:
            var_names_subset = adata.var_names.isin(gene_subset)
            subset &= var_names_subset if len(var_names_subset) > 0 else gene_subset
        elif f"{vkey}_genes" in adata.var.keys():
            subset &= np.array(adata.var[f"{vkey}_genes"].values, dtype=bool)

        xkey = xkey if xkey in adata.layers.keys() else "spliced"

        X = np.array(
            adata.layers[xkey].A[:, subset]
            if issparse(adata.layers[xkey])
            else adata.layers[xkey][:, subset]
        )
        V = np.array(
            adata.layers[vkey].A[:, subset]
            if issparse(adata.layers[vkey])
            else adata.layers[vkey][:, subset]
        )

        nans = np.isnan(np.sum(V, axis=0))
        if np.any(nans):
            X = X[:, ~nans]
            V = V[:, ~nans]

        if approx is True and X.shape[1] > 100:
            X_pca, PCs, _, _ = pca(X, n_comps=30, svd_solver="arpack", return_info=True)
            self.X = np.array(X_pca, dtype=np.float32)
            self.V = (V - V.mean(0)).dot(PCs.T)
            self.V[V.sum(1) == 0] = 0
        else:
            self.X = np.array(X, dtype=np.float32)
            self.V = np.array(V, dtype=np.float32)
        self.V_raw = np.array(self.V)

        self.sqrt_transform = sqrt_transform
        uns_key = f"{vkey}_params"
        if self.sqrt_transform is None:
            if uns_key in adata.uns.keys() and "mode" in adata.uns[uns_key]:
                self.sqrt_transform = adata.uns[uns_key]["mode"] == "stochastic"
        if self.sqrt_transform:
            self.V = np.sqrt(np.abs(self.V)) * np.sign(self.V)
        # self.V -= np.nanmean(self.V, axis=1)[:, None]

        self.n_recurse_neighbors = n_recurse_neighbors
        if self.n_recurse_neighbors is None:
            if n_neighbors is not None or mode_neighbors == "connectivities":
                self.n_recurse_neighbors = 1
            else:
                self.n_recurse_neighbors = 2

        if "neighbors" not in adata.uns.keys():
            neighbors(adata)
        if np.min((get_neighs(adata, "distances") > 0).sum(1).A1) == 0:
            raise ValueError(
                "Your neighbor graph seems to be corrupted. "
                "Consider recomputing via pp.neighbors."
            )
        if n_neighbors is None or n_neighbors <= get_n_neighs(adata):
            self.indices = get_indices(
                dist=get_neighs(adata, "distances"),
                n_neighbors=n_neighbors,
                mode_neighbors=mode_neighbors,
            )[0]
        else:
            if basis is None:
                basis_keys = ["X_pca", "X_tsne", "X_umap"]
                basis = [key for key in basis_keys if key in adata.obsm.keys()][-1]
            elif f"X_{basis}" in adata.obsm.keys():
                basis = f"X_{basis}"

            if isinstance(approx, str) and approx in adata.obsm.keys():
                from sklearn.neighbors import NearestNeighbors

                neighs = NearestNeighbors(n_neighbors=n_neighbors + 1)
                neighs.fit(adata.obsm[approx])
                self.indices = neighs.kneighbors_graph(
                    mode="connectivity"
                ).indices.reshape((-1, n_neighbors + 1))
            else:
                from scvelo import Neighbors

                neighs = Neighbors(adata)
                neighs.compute_neighbors(
                    n_neighbors=n_neighbors, use_rep=basis, n_pcs=10
                )
                self.indices = get_indices(
                    dist=neighs.distances, mode_neighbors=mode_neighbors
                )[0]

        self.max_neighs = random_neighbors_at_max

        gkey, gkey_ = f"{vkey}_graph", f"{vkey}_graph_neg"
        self.graph = adata.uns[gkey] if gkey in adata.uns.keys() else []
        self.graph_neg = adata.uns[gkey_] if gkey_ in adata.uns.keys() else []

        if tkey in adata.obs.keys():
            self.t0 = adata.obs[tkey].astype("category").copy()
            init = min(self.t0) if isinstance(min(self.t0), int) else 0
            self.t0 = self.t0.cat.set_categories(
                np.arange(init, len(self.t0.cat.categories)), rename=True
            )
            self.t1 = self.t0.copy()
            self.t1 = self.t1.cat.set_categories(
                self.t0.cat.categories + 1, rename=True
            )
        else:
            self.t0 = None

        self.compute_uncertainties = compute_uncertainties
        self.uncertainties = None
        self.self_prob = None
        self.report = report
        self.adata = adata

    # TODO: Add docstrings
    def compute_cosines(
        self, n_jobs=None, backend="loky", show_progress_bar: bool = True
    ):
        """TODO."""
        n_jobs = get_n_jobs(n_jobs=n_jobs)

        n_obs = self.X.shape[0]

        # TODO: Use batches and vectorize calculation of dX in self._calculate_cosines
        res = parallelize(
            self._compute_cosines,
            range(self.X.shape[0]),
            n_jobs=n_jobs,
            unit="cells",
            backend=backend,
            as_array=False,
            show_progress_bar=show_progress_bar,
        )()
        uncertainties, vals, rows, cols = map(_flatten, zip(*res))

        vals = np.hstack(vals)
        vals[np.isnan(vals)] = 0

        self.graph, self.graph_neg = vals_to_csr(
            vals, rows, cols, shape=(n_obs, n_obs), split_negative=True
        )
        if self.compute_uncertainties:
            uncertainties = np.hstack(uncertainties)
            uncertainties[np.isnan(uncertainties)] = 0
            self.uncertainties = vals_to_csr(
                uncertainties, rows, cols, shape=(n_obs, n_obs), split_negative=False
            )
            self.uncertainties.eliminate_zeros()

        confidence = self.graph.max(1).A.flatten()
        self.self_prob = np.clip(np.percentile(confidence, 98) - confidence, 0, 1)

    def _compute_cosines(self, obs_idx, queue):
        vals, rows, cols, uncertainties = [], [], [], []
        if self.compute_uncertainties:
            moments = get_moments(self.adata, np.sign(self.V_raw), second_order=True)

        for obs_id in obs_idx:
            if self.V[obs_id].max() != 0 or self.V[obs_id].min() != 0:
                neighs_idx = get_iterative_indices(
                    self.indices, obs_id, self.n_recurse_neighbors, self.max_neighs
                )

                if self.t0 is not None:
                    t0, t1 = self.t0[obs_id], self.t1[obs_id]
                    if t0 >= 0 and t1 > 0:
                        t1_idx = np.where(self.t0 == t1)[0]
                        if len(t1_idx) > len(neighs_idx):
                            t1_idx = np.random.choice(
                                t1_idx, len(neighs_idx), replace=False
                            )
                        if len(t1_idx) > 0:
                            neighs_idx = np.unique(np.concatenate([neighs_idx, t1_idx]))

                dX = self.X[neighs_idx] - self.X[obs_id, None]  # 60% of runtime
                if self.sqrt_transform:
                    dX = np.sqrt(np.abs(dX)) * np.sign(dX)
                val = cosine_correlation(dX, self.V[obs_id])  # 40% of runtime

                if self.compute_uncertainties:
                    dX /= l2_norm(dX)[:, None]
                    uncertainties.extend(
                        np.nansum(dX**2 * moments[obs_id][None, :], 1)
                    )

                vals.extend(val)
                rows.extend(np.ones(len(neighs_idx)) * obs_id)
                cols.extend(neighs_idx)

            if queue is not None:
                queue.put(1)

        if queue is not None:
            queue.put(None)

        return uncertainties, vals, rows, cols


def _flatten(iterable):
    return [i for it in iterable for i in it]


def velocity_graph(
    data,
    vkey="velocity",
    xkey="Ms",
    tkey=None,
    basis=None,
    n_neighbors=None,
    n_recurse_neighbors=None,
    random_neighbors_at_max=None,
    sqrt_transform=None,
    variance_stabilization=None,
    gene_subset=None,
    compute_uncertainties=None,
    approx=None,
    mode_neighbors="distances",
    copy=False,
    n_jobs=None,
    backend="loky",
    show_progress_bar: bool = True,
):
    r"""Computes velocity graph based on cosine similarities.

    The cosine similarities are computed between velocities and potential cell state
    transitions, i.e. it measures how well a corresponding change in gene expression
    :math:`\delta_{ij} = x_j - x_i` matches the predicted change according to the
    velocity vector :math:`\nu_i`,

    .. math::
        \pi_{ij} = \cos\angle(\delta_{ij}, \nu_i)
        = \frac{\delta_{ij}^T \nu_i}{\left\lVert\delta_{ij}\right\rVert
        \left\lVert \nu_i \right\rVert}.

    Arguments:
    ---------
    data: :class:`~anndata.AnnData`
        Annotated data matrix.
    vkey: `str` (default: `'velocity'`)
        Name of velocity estimates to be used.
    xkey: `str` (default: `'Ms'`)
        Layer key to extract count data from.
    tkey: `str` (default: `None`)
        Observation key to extract time data from.
    basis: `str` (default: `None`)
        Basis / Embedding to use.
    n_neighbors: `int` or `None` (default: None)
        Use fixed number of neighbors or do recursive neighbor search (if `None`).
    n_recurse_neighbors: `int` (default: `None`)
        Number of recursions for neighbors search. Defaults to
        2 if mode_neighbors is 'distances', and 1 if mode_neighbors is 'connectivities'.
    random_neighbors_at_max: `int` or `None` (default: `None`)
        If number of iterative neighbors for an individual cell is higher than this
        threshold, a random selection of such are chosen as reference neighbors.
    sqrt_transform: `bool` (default: `False`)
        Whether to variance-transform the cell states changes
        and velocities before computing cosine similarities.
    gene_subset: `list` of `str`, subset of adata.var_names or `None`(default: `None`)
        Subset of genes to compute velocity graph on exclusively.
    compute_uncertainties: `bool` (default: `None`)
        Whether to compute uncertainties along with cosine correlation.
    approx: `bool` or `None` (default: `None`)
        If True, first 30 pc's are used instead of the full count matrix
    mode_neighbors: 'str' (default: `'distances'`)
        Determines the type of KNN graph used. Options are 'distances' or
        'connectivities'. The latter yields a symmetric graph.
    copy: `bool` (default: `False`)
        Return a copy instead of writing to adata.
    n_jobs: `int` or `None` (default: `None`)
        Number of parallel jobs.
    backend: `str` (default: "loky")
        Backend used for multiprocessing. See :class:`joblib.Parallel` for valid
        options.
    show_progress_bar
        Whether to show a progress bar.

    Returns
    -------
    velocity_graph: `.uns`
        sparse matrix with correlations of cell state transitions with velocities
    """
    adata = data.copy() if copy else data
    verify_neighbors(adata)
    if vkey not in adata.layers.keys():
        velocity(adata, vkey=vkey)
    if sqrt_transform is None:
        sqrt_transform = variance_stabilization

    vgraph = VelocityGraph(
        adata,
        vkey=vkey,
        xkey=xkey,
        tkey=tkey,
        basis=basis,
        n_neighbors=n_neighbors,
        approx=approx,
        n_recurse_neighbors=n_recurse_neighbors,
        random_neighbors_at_max=random_neighbors_at_max,
        sqrt_transform=sqrt_transform,
        gene_subset=gene_subset,
        compute_uncertainties=compute_uncertainties,
        report=True,
        mode_neighbors=mode_neighbors,
    )

    if isinstance(basis, str):
        logg.warn(
            f"The velocity graph is computed on {basis} embedding coordinates.\n"
            f"        Consider computing the graph in an unbiased manner \n"
            f"        on full expression space by not specifying basis.\n"
        )

    n_jobs = get_n_jobs(n_jobs=n_jobs)
    logg.info(
        f"computing velocity graph (using {n_jobs}/{os.cpu_count()} cores)", r=True
    )
    vgraph.compute_cosines(
        n_jobs=n_jobs, backend=backend, show_progress_bar=show_progress_bar
    )

    adata.uns[f"{vkey}_graph"] = vgraph.graph
    adata.uns[f"{vkey}_graph_neg"] = vgraph.graph_neg

    if vgraph.uncertainties is not None:
        adata.uns[f"{vkey}_graph_uncertainties"] = vgraph.uncertainties

    adata.obs[f"{vkey}_self_transition"] = vgraph.self_prob

    if f"{vkey}_params" in adata.uns.keys():
        if "embeddings" in adata.uns[f"{vkey}_params"]:
            del adata.uns[f"{vkey}_params"]["embeddings"]
    else:
        adata.uns[f"{vkey}_params"] = {}
    adata.uns[f"{vkey}_params"]["mode_neighbors"] = mode_neighbors
    adata.uns[f"{vkey}_params"]["n_recurse_neighbors"] = vgraph.n_recurse_neighbors

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        f"    '{vkey}_graph', sparse matrix with cosine correlations (adata.uns)"
    )

    return adata if copy else None

def coar_adj(result_adata, pyg_data, path, name):
    out_adj = np.matmul(np.matmul(result_adata.obsm['cluster_matrix'].T, pyg_data.adj.cpu()), result_adata.obsm['cluster_matrix'])
    # Plotting the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(out_adj, annot=True, cmap='viridis')
    plt.title('Coarsened Adjacency Matrix Heatmap')
    plt.xlabel('Cluster Index')
    plt.ylabel('Cluster Index')
    plt.savefig(path + name + ' clusters_coar_adj.svg')

def plot_stacked_bar(adata, bar_element, stack_element, path, stack_order=None, bar_order=None, ax=None, size=(5, 4)):
    """
    Plots a stacked bar plot from an AnnData object with bars ordered based on the provided bar_order or 
    the mean of 'Pred Time', and stacks ordered based on a provided stack_order, while maintaining the color 
    alignment using a color dictionary.

    Parameters:
    adata: AnnData
        The annotated data matrix.
    bar_element: str
        The category to use for the bars.
    stack_element: str
        The category to use for the stacked elements within each bar.
    path: str
        The directory where the plot will be saved.
    stack_order: list, optional
        The order of the categories for the stacked elements.
    bar_order: list, optional
        The order of the bars (bar_element categories). If not provided, the bars are ordered by mean 'Pred Time'.
    ax: matplotlib.axes.Axes, optional
        Matplotlib axes to plot on. If not provided, a new figure and axes will be created.
    """
    # Ensure the bar_element and stack_element are treated as categorical columns
    adata.obs[bar_element] = adata.obs[bar_element].astype('category')
    adata.obs[stack_element] = adata.obs[stack_element].astype('category')
    
    # If bar_order is provided, use it to reorder the bars
    if bar_order is not None:
        adata.obs[bar_element] = adata.obs[bar_element].cat.reorder_categories(bar_order, ordered=True)
    else:
        # Calculate the mean of 'Pred Time' for each category in bar_element
        mean_pred_time = adata.obs.groupby(bar_element)['Pred Time'].mean()
        # Sort the bar_element categories based on the mean of 'Pred Time'
        sorted_categories = mean_pred_time.sort_values().index.tolist()
        # Reorder the categories of bar_element according to the sorted means
        adata.obs[bar_element] = adata.obs[bar_element].cat.reorder_categories(sorted_categories, ordered=True)
    
    # Reorder the stack_element categories if a specific order is provided
    if stack_order is not None:
        adata.obs[stack_element] = adata.obs[stack_element].cat.reorder_categories(stack_order, ordered=True)
        
        # Use the dictionary of colors from adata.uns to reorder the colors
        color_dict = adata.uns.get(stack_element+'_colors', {})
        
        # Reorder the colors based on the new stack_order
        reordered_colors = [color_dict[elem] for elem in stack_order if elem in color_dict]
    else:
        # Use a colormap to generate colors if no specific stack order is provided
        n_colors = len(adata.obs[stack_element].cat.categories)
        cmap = cm.get_cmap('viridis', n_colors)  # You can replace 'viridis' with any colormap you prefer
        reordered_colors = [cmap(i) for i in np.linspace(0, 1, n_colors)]
    
    # Group by the specified elements and calculate counts
    element_counts = adata.obs.groupby([bar_element, stack_element]).size().unstack(fill_value=0)
    
    # Calculate proportions
    element_proportions = element_counts.div(element_counts.sum(axis=1), axis=0)

    # Plotting the stacked bar plot, using the passed ax if available
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
        save_figure = True  # Flag to save the figure if ax is None
    else:
        save_figure = False  # Don't save the figure if ax is provided
    
    element_proportions.plot(kind='bar', stacked=True, ax=ax, color=reordered_colors)
    
    # Add labels and title
    ax.set_ylabel('Proportion')
    ax.set_xlabel(bar_element.capitalize())
    ax.set_title(f'{stack_element.capitalize()} Proportions by {bar_element.capitalize()}')
    ax.legend(title=stack_element.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if save_figure:
        # Save the figure
        plt.tight_layout()
        plt.savefig(path + 'stack_by_'+bar_element+'_'+stack_element+'.pdf')
        plt.show()

    # Return the ax for further customization if needed
    return ax

def plot_sankey_from_anndata(
    adata, 
    left_label_col, 
    right_label_col, 
    title="Sankey Diagram", 
    output_path=None, 
    left_label_order=None, 
    right_label_order=None,
    left_colors=None,      # New parameter for left label colors
    right_colors=None      # New parameter for right label colors
):
    """
    Creates a Sankey diagram based on two categorical labels from an AnnData object using Plotly, 
    with flows colored by the labels defined in `adata.uns` for either the left or right labels.
    
    Parameters:
    adata: AnnData
        The annotated data matrix (must have the left_label_col and right_label_col in obs).
    left_label_col: str
        The name of the column in adata.obs to use as the left labels (e.g., celltype).
    right_label_col: str
        The name of the column in adata.obs to use as the right labels (e.g., cluster).
    title: str, optional
        Title of the Sankey diagram.
    output_path: str, optional
        Path to save the figure as an SVG file. If None, the figure won't be saved.
    left_label_order: list, optional
        Custom order for the left labels. If None, the order is determined by the unique values in the data.
    right_label_order: list, optional
        Custom order for the right labels. If None, the order is determined by the unique values in the data.
    left_colors: list, optional
        List of colors for the left labels, in the same order as left_label_order.
        If None, colors are retrieved from `adata.uns` or default colors are used.
    right_colors: list, optional
        List of colors for the right labels, in the same order as right_label_order.
        If None, colors are retrieved from `adata.uns` or default colors are used.
        
    Returns:
    A Plotly Figure object representing the Sankey diagram.
    """
    import pandas as pd
    import plotly.graph_objects as go
    import os

    # Helper function to convert hex colors to rgba with opacity
    def hex_to_rgba(hex_color, opacity):
        """Convert hex color to rgba color with the given opacity."""
        hex_color = hex_color.lstrip('#')
        hlen = len(hex_color)
        rgb = tuple(int(hex_color[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))
        return f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{opacity})'

    # Extract left and right labels from adata.obs and ensure they are strings
    left_labels = adata.obs[left_label_col].astype(str)
    right_labels = adata.obs[right_label_col].astype(str)

    # Define unique labels in the specified order
    unique_left_labels = left_label_order if left_label_order else left_labels.unique().tolist()
    unique_right_labels = right_label_order if right_label_order else right_labels.unique().tolist()

    # Validate that the provided color lists match the label orders
    if left_colors:
        if len(left_colors) != len(unique_left_labels):
            raise ValueError("Length of left_colors does not match the number of left labels.")
    if right_colors:
        if len(right_colors) != len(unique_right_labels):
            raise ValueError("Length of right_colors does not match the number of right labels.")

    # Create a DataFrame to group and count transitions between left_label_col and right_label_col
    df = pd.DataFrame({
        left_label_col: left_labels,
        right_label_col: right_labels
    })

    # Group by the two labels to get counts (the "flow" values for the Sankey diagram)
    flow_data = df.groupby([left_label_col, right_label_col]).size().reset_index(name='count')

    # Assign unique indices to the left and right labels
    left_label_index_map = {label: idx for idx, label in enumerate(unique_left_labels)}
    right_label_index_map = {label: idx + len(unique_left_labels) for idx, label in enumerate(unique_right_labels)}

    # Prepare source, target, and value lists for the Sankey diagram
    flow_data[left_label_col] = flow_data[left_label_col].astype(str)
    flow_data[right_label_col] = flow_data[right_label_col].astype(str)
    
    source = flow_data[left_label_col].map(left_label_index_map).tolist()
    target = flow_data[right_label_col].map(right_label_index_map).tolist()
    value = flow_data['count'].tolist()

    # Combine both unique lists for labels
    labels = unique_left_labels + unique_right_labels

    # If colors are not provided, retrieve from adata.uns or use default colors
    if not left_colors:
        # Retrieve colors from `adata.uns` if available
        left_colors_in_data = adata.uns.get(f'{left_label_col}_colors', ['#1f77b4'] * len(unique_left_labels))
        # Handle cases where colors might be stored as dictionaries
        if isinstance(left_colors_in_data, dict):
            left_colors_in_data = [left_colors_in_data.get(label, '#1f77b4') for label in unique_left_labels]
        left_colors = left_colors_in_data  # Assign to left_colors

    if not right_colors:
        # Retrieve colors from `adata.uns` if available
        right_colors_in_data = adata.uns.get(f'{right_label_col}_colors', ['#D3D3D3'] * len(unique_right_labels))
        # Handle cases where colors might be stored as dictionaries
        if isinstance(right_colors_in_data, dict):
            right_colors_in_data = [right_colors_in_data.get(label, '#D3D3D3') for label in unique_right_labels]
        right_colors = right_colors_in_data  # Assign to right_colors

    # Combine node colors
    node_colors = left_colors + right_colors

    # Create link colors based on source node's color with opacity
    link_colors = [hex_to_rgba(left_colors[src], 0.5) for src in source]

    # Create Sankey diagram using Plotly
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",  # Better control over node positions
        node=dict(
            pad=10,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors,
            hovertemplate='%{label}<extra></extra>'
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors,
            hovertemplate='Source: %{source.label}<br />Target: %{target.label}<br />Value: %{value}<extra></extra>',
            line=dict(color='rgba(0,0,0,0)', width=0.5)  # Remove borders around links
        )
    )])
    # Set the title and adjust layout
    fig.update_layout(
        title_text=title,
        font_size=7,
        title_font_size=7,
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=300,   # Set figure width to match Nature Methods guidelines
        height=300,  # Adjust height proportionally
        margin=dict(l=10, r=10, t=40, b=10),  # Reduce margins for compactness
    )

    # Ensure fonts and sizes are appropriate for publication
    fig.update_layout(
        font=dict(
            family='Arial',
            size=7,  # Font size suitable for print (~7-9 pt)
            color='black'
        )
    )

    # Render the Sankey diagram
    if output_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save as SVG using kaleido
        fig.write_image(output_path, format='svg')
        print(f"Sankey diagram saved to {output_path}")

    return fig

def plot_curve(adata, gene_of_interest, savepath, smooth=70):

    # Replace 'gene_of_interest' with the gene name or index you are interested in
    gene_idx = adata.var_names.get_loc(gene_of_interest)

    # Step 2: Extract pred_time from obs
    time_points = adata.obs['Pred Time'].values
    celltype = adata.obs['celltype'].values
    expert = adata.obs['Expert'].values

    # Step 3: Extract data from the respective layers for the gene of interest
    Mc_values = adata.layers['Mc'].toarray()[:, gene_idx]
    recon_alpha_values = adata.layers['recon_alpha_norm'][:, gene_idx]
    Mu_values = adata.layers['scale_Mu'][:, gene_idx]
    Ms_values = adata.layers['scale_Ms'][:, gene_idx]

    # Step 4: Create a dataframe to easily plot these values
    df = pd.DataFrame({
        'Pred Time': time_points,
        'C': Mc_values,
        'Pred_Alpha': recon_alpha_values,
        'U': Mu_values,
        'S': Ms_values,
        'Celltype': celltype,
        'Pred_Expert': expert,
    })

    # Step 5: Group by pred_time and compute the mean for each group
    df_mean = df.groupby('Pred Time').agg({
        'C': 'mean',
        'Pred_Alpha': 'mean',
        'U': 'mean',
        'S': 'mean',
        'Celltype': lambda x: x.mode()[0],  # Use the most frequent celltype
        'Pred_Expert': lambda x: x.mode()[0]  # Use the most frequent celltype
    }).reset_index()

    # Define a simple averaging kernel (for smoothing)
    kernel = np.ones(smooth) / smooth  # Adjust kernel size for more/less smoothing

    # Step 6: Apply np.convolve for smoothing
    df_mean['C'] = np.convolve(df_mean['C'], kernel, mode='same')
    df_mean['Pred_Alpha'] = np.convolve(df_mean['Pred_Alpha'], kernel, mode='same')
    df_mean['U'] = np.convolve(df_mean['U'], kernel, mode='same')
    df_mean['S'] = np.convolve(df_mean['S'], kernel, mode='same')

    # Step 7: Normalize each smoothed column to [0, 1] range
    def min_max_normalize(series):
        return (series - series.min()) / (series.max() - series.min())

    df_mean['C'] = min_max_normalize(df_mean['C']) * 1.5
    df_mean['Pred_Alpha'] = min_max_normalize(df_mean['Pred_Alpha'])
    df_mean['U'] = min_max_normalize(df_mean['U'])
    df_mean['S'] = min_max_normalize(df_mean['S'])

    # Step 8: Compute the difference (gradient) for each smoothed column
    df_mean['C_diff'] = np.abs(np.gradient(df_mean['C']))
    df_mean['Pred_Alpha_diff'] = np.abs(np.gradient(df_mean['Pred_Alpha']))
    df_mean['U_diff'] = np.abs(np.gradient(df_mean['U']))
    df_mean['S_diff'] = np.abs(np.gradient(df_mean['S']))

    # Step 1: Melt the DataFrame into long format
    df_long = pd.melt(
        df_mean,
        id_vars=['Pred Time', 'Celltype', 'Pred_Expert'],  # Columns to keep
        value_vars=['C', 'Pred_Alpha', 'U', 'S'],  # Columns to melt
        var_name='Variable',  # Name for the new column that will hold the variable names
        value_name='Normalized_Expression'  # Name for the new column that will hold the values
    )

    # Initialize a figure for the subplots
    fig, axes = plt.subplots(2, 1, figsize=(5, 9), sharex=True, sharey=True)

    # First plot: Hue based on 'Variable'
    sns.scatterplot(
        data=df_long,
        x='Pred Time',
        y='Normalized_Expression',
        hue='Variable',  # Different colors for each variable
        edgecolor='none',
        s=4,
        legend='brief',
        ax=axes[0]  # Plot on the first subplot
    )
    axes[0].set_title('Hue by Variable')
    axes[0].set_xlabel('Predicted Time')
    axes[0].set_ylabel('Normalized Expression Levels')

    # Second plot: Hue based on 'Pred_Expert'
    sns.scatterplot(
        data=df_long,
        x='Pred Time',
        y='Normalized_Expression',
        hue='Pred_Expert',  # Different colors for each expert prediction
        edgecolor='none',
        palette='turbo',
        s=4,
        legend='brief',
        ax=axes[1]  # Plot on the second subplot
    )
    axes[1].set_title('Hue by Pred_Expert')
    axes[1].set_xlabel('Predicted Time')
    axes[1].set_ylabel('Normalized Expression Levels')

    # Set the overall title for the figure
    fig.suptitle(f'Total Trends of C, Alpha, U, and S for {gene_of_interest}')
    plt.tight_layout()
    plt.savefig(savepath + gene_of_interest + '.svg')

def expert_DG(adata, expert1, expert2,savepath,layer=None,obs_key='Expert',size=(5, 4)):
    adata_subset = adata[adata.obs[obs_key].isin([expert1, expert2])].copy()
    sc.tl.rank_genes_groups(adata_subset, obs_key,layer=layer, groups=[expert1, expert2])
    result = adata_subset.uns['rank_genes_groups']
    genes = result['names'][expert1]
    logfoldchanges = result['logfoldchanges'][expert1]
    pvals = result['pvals'][expert1]
    volcano_data = pd.DataFrame({
        'gene': genes,
        'logFC': logfoldchanges,
        '-log10(p-value)': -np.log10(pvals)
    })
    significance_cutoff = 0.05
    logfc_cutoff = 1
    plt.figure(figsize=size)
    plt.scatter(volcano_data['logFC'], volcano_data['-log10(p-value)'], color='grey',s=10, label='Not Significant')
    significant_up = volcano_data[(volcano_data['logFC'] > logfc_cutoff) & (volcano_data['-log10(p-value)'] > -np.log10(significance_cutoff))]
    significant_down = volcano_data[(volcano_data['logFC'] < -logfc_cutoff) & (volcano_data['-log10(p-value)'] > -np.log10(significance_cutoff))]
    # Plot significant upregulated genes (red) and downregulated genes (blue)
    plt.scatter(significant_up['logFC'], significant_up['-log10(p-value)'], color='red',s=10, label='Upregulated in ' + expert1)
    plt.scatter(significant_down['logFC'], significant_down['-log10(p-value)'], color='blue',s=10, label='Upregulated in ' + expert2)
    # Add lines for thresholds
    plt.axhline(y=-np.log10(significance_cutoff), color='black', linestyle='--', lw=1)
    plt.axvline(x=logfc_cutoff, color='black', linestyle='--', lw=1)
    plt.axvline(x=-logfc_cutoff, color='black', linestyle='--', lw=1)
    # Add plot labels and title
    plt.xlabel('Log2 Fold Change', fontsize=7)
    plt.ylabel('-Log10(p-value)', fontsize=7)
    plt.title('Volcano Plot: '+expert1+' vs '+expert2, fontsize=7)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if layer:
        plt.savefig(savepath + obs_key+'_DG_'+expert1+'_'+expert2+'_'+layer+'.pdf')
    else:
        plt.savefig(savepath + obs_key+'_DG_'+expert1+'_'+expert2+'.pdf')
    # Export to CSV for further analysis in R
    marker_genes_1 = pd.DataFrame(significant_up['gene'])  # Group 5
    marker_genes_2 = pd.DataFrame(significant_down['gene']) # Group 3
    if layer:
        marker_genes_1.to_csv(savepath+obs_key+'_'+expert1+'_'+layer+'_markers.csv', index=False, header=False)
        marker_genes_2.to_csv(savepath+obs_key+'_'+expert2+'_'+layer+'_markers.csv', index=False, header=False)
    else:
        marker_genes_1.to_csv(savepath+obs_key+'_'+expert1+'_markers.csv', index=False, header=False)
        marker_genes_2.to_csv(savepath+obs_key+'_'+expert2+'_markers.csv', index=False, header=False)
    return marker_genes_1, marker_genes_2

def expert_DG_multi(adata, cluster_list, key='Expert',p_key='pvals',significance_cutoff=0.05, logfc_cutoff=1):
    
    # Subset the data to include only the specified clusters
    adata_subset = adata[adata.obs[key].isin(cluster_list)].copy()
    
    # Run differential expression analysis
    sc.tl.rank_genes_groups(adata_subset, groupby=key)
    
    # Extract the results from adata.uns
    results = adata_subset.uns['rank_genes_groups']
    groups = results['names'].dtype.names
# Initialize an empty DataFrame to store all the differential genes
    differential_genes_list = []

    # Loop through each group and extract the relevant data
    for group in groups:
        group_df = pd.DataFrame({
            'gene': results['names'][group],
            'logfoldchange': results['logfoldchanges'][group],
            'pvals': results['pvals'][group],
            'pvals_adj': results['pvals_adj'][group],
            'group': group
        })
        
        # Append this group's DataFrame to the list
        differential_genes_list.append(group_df)

    # Concatenate all dataframes into one
    differential_genes = pd.concat(differential_genes_list, ignore_index=True)

    # Convert the columns to appropriate data types
    differential_genes['logfoldchange'] = pd.to_numeric(differential_genes['logfoldchange'], errors='coerce')
    differential_genes['pvals'] = pd.to_numeric(differential_genes['pvals'], errors='coerce')
    differential_genes['pvals_adj'] = pd.to_numeric(differential_genes['pvals_adj'], errors='coerce')

    # Filter genes based on log fold change and adjusted p-value thresholds
    significant_genes = differential_genes[
        (differential_genes['logfoldchange'].abs() > logfc_cutoff) & (differential_genes[p_key] < significance_cutoff)
    ]

    # Create a dictionary to store the top n genes for each cluster
    top_genes = {}
    for group in cluster_list:
        # Select genes for the current group and get the top 'n' genes
        top_genes_group = significant_genes[significant_genes['group'] == group]
        top_genes[group] = top_genes_group['gene'].tolist()

    return top_genes

def get_all_significant_genes(adata,layer, cluster_list, savepath, pvalue_cutoff=0.05, logfc_cutoff=1):
    """
    Identifies all significant genes for each cluster based on given p-value and log fold change cutoffs,
    and saves the genes in separate DataFrames for each cluster.

    Parameters:
    -----------
    adata : AnnData
        The AnnData object containing the data.
    cluster_list : list
        List of clusters to be analyzed (values should match those in adata.obs['clusters']).
    pvalue_cutoff : float
        p-value cutoff to consider genes significant.
    logfc_cutoff : float
        Log fold change cutoff to consider genes as differentially expressed.

    Returns:
    --------
    cluster_gene_dfs : dict
        A dictionary with clusters as keys and a DataFrame of the significant genes for each cluster.
    """

    # Subset the data to include only the specified clusters
    adata_subset = adata[adata.obs['Expert'].isin(cluster_list)].copy()
    
    # Run differential expression analysis
    sc.tl.rank_genes_groups(adata_subset, groupby='Expert', layer= layer)
    
    # Extract the results from adata.uns
    results = adata_subset.uns['rank_genes_groups']
    groups = results['names'].dtype.names

    # Initialize an empty dictionary to store DataFrames for each cluster
    cluster_gene_dfs = {}

    # Loop through each group/cluster and extract the relevant data
    for group in groups:
        group_df = pd.DataFrame({
            'gene': results['names'][group],
            'logfoldchange': results['logfoldchanges'][group],
            'pvals': results['pvals'][group],
            'pvals_adj': results['pvals_adj'][group],
            'group': group
        })

        # Convert columns to numeric types for proper filtering
        group_df['logfoldchange'] = pd.to_numeric(group_df['logfoldchange'], errors='coerce')
        group_df['pvals_adj'] = pd.to_numeric(group_df['pvals_adj'], errors='coerce')

        # Filter based on the given thresholds (using p-value instead of p-value adjusted)
        filtered_genes = group_df[
            (group_df['logfoldchange'] > logfc_cutoff) & (group_df['pvals_adj'] < pvalue_cutoff)
        ]
        
        # Save the filtered DataFrame to the dictionary
        cluster_gene_dfs[group] = filtered_genes.reset_index(drop=True)

    # Save each cluster's DataFrame into separate files or print them
    for cluster, df in cluster_gene_dfs.items():
        print(f"Significant genes for {cluster}:")
        print(df)
        # Save to a CSV file if needed
        df.to_csv(f"{savepath}Expert_{cluster}_up_expression_genes.csv", index=False)

    return cluster_gene_dfs


# Preprocessing function
def preprocess_dtw(adata, gene, para, time_key, bins):
    # Get values
    arr = np.array(adata[:, gene].layers[para].toarray())[:, 0] if issparse(adata[:, gene].layers[para]) else np.array(adata[:, gene].layers[para])[:, 0]
    # Get time data
    time = np.array(adata[:, gene].layers[time_key].toarray())[:, 0] if issparse(adata[:, gene].layers[time_key]) else np.array(adata[:, gene].layers[time_key])[:, 0]
    # Normalize the time data
    new_time = (time - time.min()) / (time.max() - time.min())
    # Convert time data into integer bins
    arr_window = np.floor(new_time * bins)
    # Group data into time bins and take the mean
    df = DataFrame({'arr': arr, 'window': arr_window})
    arr_df = df.groupby(['window'], group_keys=False).mean()
    # Binned data and corresponding time bins
    arr2 = np.array(arr_df['arr'])
    arr_t = np.linspace(0, 1, len(arr2))
    # Normalize the data
    arr_vec = arr2 - min(arr2)
    arr_vec = arr_vec / max(arr_vec)
    return arr_t, arr_vec

# Function to plot DTW alignment with alignment lines
def plot_dtw_alignment(t1, s1, t2, s2, aligned, ax, label1, label2,color1,color2, title):
    """
    Plots DTW alignment between two sequences with alignment lines.

    Parameters:
    - t1, s1: Time and signal for the first sequence.
    - t2, s2: Time and signal for the second sequence.
    - aligned: DTW alignment object.
    - ax: Matplotlib axes object to plot on.
    - label1, label2: Labels for the sequences.
    - title: Title of the subplot.
    """
    ax.plot(t1, s1, label=label1, color=color1, linestyle='-',linewidth=2)
    ax.plot(t2, s2, label=label2, color=color2, linestyle='--',linewidth=2)
    for i in range(len(aligned.index1)):
        ax.plot([t1[aligned.index1[i]], t2[aligned.index2[i]]],
                [s1[aligned.index1[i]], s2[aligned.index2[i]]],
                color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Pred Time')
    ax.set_ylabel('Normalized Values')
    ax.legend()
    ax.set_title(title)


# Calculate lags between aligned sequences
def calculate_lag(aligned, t1, t2):
    """
    Calculates lag between two aligned sequences.

    Parameters:
    - aligned: DTW alignment object.
    - t1, t2: Time arrays for the two sequences.

    Returns:
    - time_aligned: Average time of aligned points.
    - lag: Lag between the sequences.
    """
    idx1 = aligned.index1
    idx2 = aligned.index2
    t1_aligned = t1[idx1]
    t2_aligned = t2[idx2]
    lag = t2_aligned - t1_aligned
    time_aligned = (t1_aligned + t2_aligned) / 2
    return time_aligned, lag

def plot_entropy(Entropy_df, path, key='Entropy_pred_cluster'):
    # Extract the 'Entropy_pred_cluster' from result_adata.var
    entropy_values = Entropy_df[key]
    # Sort the values by 'Entropy_pred_cluster'
    sorted_entropy_values = entropy_values.sort_values()
    # Create an order based on the sorted index of the genes
    gene_order = range(len(sorted_entropy_values))
    # Now plot using matplotlib
    plt.figure(figsize=(4, 4))
    plt.scatter(gene_order, sorted_entropy_values, alpha=0.6)
    plt.title('Ordered Entropy values for Genes')
    plt.xlabel('Gene Order (Sorted by Entropy based on pred cluster)')
    plt.ylabel('Values')
    plt.tight_layout()
    plt.savefig(path + 'entropy.svg')


def plot_gene_expression(
    adata,
    genes,
    color_by,
    path,
    ncols=None,
    velocity=True,
    agg='mean',
    x_key='model_Ms',
    y_key='model_Mu',
    pred_vs_key='pred_vs',
    pred_vu_key='pred_vu',
    point_size=30,
    arrow_scale=0.5,   # Scale factor for arrow lengths
    arrow_step=1,      # Step size for downsampling arrows
    grid_size=20,
    sigma=3,
    lowerb=70,
    upperb=100,
    xlab='Spliced',
    ylab='Unspliced',
    random_seed=None,
    max_arrow_length=40,
    alpha=0.1
):
    """
    Plot gene expression data with optional velocity arrows using grid-based sampling.

    Parameters:
    -----------
    adata : AnnData
        The annotated data matrix.
    genes : list of str
        List of gene names to plot.
    color_by : str
        The observation (cell) annotation to color by.
    path : str
        File path to save the generated plot.
    ncols : int, optional
        Number of columns in the subplot grid.
    velocity : bool, optional
        Whether to plot velocity arrows. Default is True.
    agg : str, optional
        Aggregation method for grid sampling ('mean' or 'random'). Default is 'mean'.
    x_key : str, optional
        Layer key for spliced counts. Default is 'splice'.
    y_key : str, optional
        Layer key for unspliced counts. Default is 'unsplice'.
    pred_vs_key : str, optional
        Layer key for predicted spliced velocity. Default is 'pred_vs'.
    pred_vu_key : str, optional
        Layer key for predicted unspliced velocity. Default is 'pred_vu'.
    point_size : int, optional
        Size of scatter plot points. Default is 40.
    arrow_scale : float, optional
        Scale factor for arrow lengths. Default is 0.5.
    arrow_step : int, optional
        Step size for downsampling arrows. Higher value means fewer arrows. Default is 2.
    grid_size : int, optional
        Number of grid cells along each axis for sampling. Default is 25.
    sigma : float, optional
        Standard deviation for Gaussian smoothing of density. Default is 0.1.
    lowerb : float, optional
        Lower percentile threshold for density filtering. Default is 0.
    upperb : float, optional
        Upper percentile threshold for density filtering. Default is 98.
    xlab : str, optional
        Label for the x-axis. Default is 'Spliced'.
    ylab : str, optional
        Label for the y-axis. Default is 'Unspliced'.
    random_seed : int, optional
        Seed for random number generator for reproducibility. Default is None.
    """

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Determine the layout for the subplots
    n_genes = len(genes)
    if ncols is None:
        ncols = int(np.ceil(np.sqrt(n_genes)))
    nrows = int(np.ceil(n_genes / ncols))
    present_genes = [gene for gene in genes if gene in adata.var_names]
    absent_genes = [gene for gene in genes if gene not in adata.var_names]

    # Notify about absent genes
    if absent_genes:
        print(f"The following genes are not found in 'var_names': {', '.join(absent_genes)}")

    # Filter the anndata object for the selected genes
    gene_indices = [adata.var_names.get_loc(gene) for gene in present_genes]
    splice = adata.layers[x_key][:, gene_indices]
    unsplice = adata.layers[y_key][:, gene_indices]
    pred_vs = adata.layers[pred_vs_key][:, gene_indices]
    pred_vu = adata.layers[pred_vu_key][:, gene_indices]

    # Extract colors from the obs column
    color_categories = adata.obs[color_by].astype('category')
    color_codes = color_categories.cat.codes

    # Check if a colormap is available in adata.uns
    cmap_key = f"{color_by}_colors"
    if cmap_key in adata.uns:
        color_map = adata.uns[cmap_key]
        colors = [color_map[i] for i in color_codes]
        legend_elements = [
            mpatches.Patch(color=color_map[i], label=category)
            for i, category in enumerate(color_categories.cat.categories)
        ]
    else:
        plt_cmap = 'viridis'
        colors = color_codes
        unique_codes = np.unique(color_codes)
        legend_elements = [
            mpatches.Patch(
                color=plt.get_cmap(plt_cmap)(code / len(unique_codes)),
                label=category
            )
            for code, category in zip(unique_codes, color_categories.cat.categories)
        ]

    # Create figure and axes for subplots
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, 3 * nrows))
    axs = axs.flatten() if n_genes > 1 else [axs]

    # Helper Function to Plot Velocity Arrows
    def plot_velocity_arrows(
        ax,
        splice_vals,
        unsplice_vals,
        pred_vs_vals,
        pred_vu_vals,
        grid_size=25,
        sigma=0.1,
        lowerb=0,
        upperb=98,
        agg='mean',
        arrow_scale=1.0,
        arrow_step=1,
        arrow_color='black',
        arrow_width=0.005  # Increased arrow width
    ):
        """
        Calculate and plot velocity arrows using quiver on the grid-based sampling.
        """
        # Define the grid over the data range
        x_min, x_max = splice_vals.min(), splice_vals.max()
        y_min, y_max = unsplice_vals.min(), unsplice_vals.max()

        x_bins = np.linspace(x_min, x_max, grid_size + 1)
        y_bins = np.linspace(y_min, y_max, grid_size + 1)

        # Digitize the data to assign each point to a grid cell
        x_digitized = np.digitize(splice_vals, bins=x_bins) - 1
        y_digitized = np.digitize(unsplice_vals, bins=y_bins) - 1

        # Ensure indices are within valid range
        x_digitized = np.clip(x_digitized, 0, grid_size - 1)
        y_digitized = np.clip(y_digitized, 0, grid_size - 1)

        # Compute the 2D histogram counts
        counts, _, _ = np.histogram2d(
            splice_vals, unsplice_vals, bins=[x_bins, y_bins]
        )

        # Flatten the counts to get density estimates
        density_estimate = counts.flatten()

        # Compute the density threshold
        lower_density_threshold = np.percentile(density_estimate, lowerb)
        upper_density_threshold = np.percentile(density_estimate, upperb)
        # Create a boolean mask of grid cells to keep
        bool_density = (density_estimate > lower_density_threshold) & (density_estimate < upper_density_threshold)

        # Reshape bool_density back to 2D grid
        bool_density_2d = bool_density.reshape(counts.shape)

        # Map grid cell indices to data point indices
        grid_cell_indices = x_digitized + y_digitized * grid_size

        cell_to_indices = defaultdict(list)
        for idx, grid_idx in enumerate(grid_cell_indices):
            cell_to_indices[grid_idx].append(idx)

        # Get grid cell indices that pass the density threshold
        passing_cells = np.argwhere(bool_density_2d)
        passing_x_indices = passing_cells[:, 0]
        passing_y_indices = passing_cells[:, 1]
        grid_cell_indices_passing = passing_x_indices + passing_y_indices * grid_size

        # Initialize arrays for selected data
        selected_splice = []
        selected_unsplice = []
        selected_vs = []
        selected_vu = []

        # For each passing grid cell, compute aggregated velocities
        for grid_idx in grid_cell_indices_passing:
            if grid_idx in cell_to_indices:
                indices_in_cell = cell_to_indices[grid_idx]
                if len(indices_in_cell) == 0:
                    continue  # Skip empty cells

                if agg == 'mean':
                    splice_mean = splice_vals[indices_in_cell].mean()
                    unsplice_mean = unsplice_vals[indices_in_cell].mean()
                    vs_mean = pred_vs_vals[indices_in_cell].mean()
                    vu_mean = pred_vu_vals[indices_in_cell].mean()

                    selected_splice.append(splice_mean)
                    selected_unsplice.append(unsplice_mean)
                    selected_vs.append(vs_mean)
                    selected_vu.append(vu_mean)
                elif agg == 'random':
                    selected_idx = np.random.choice(indices_in_cell)
                    selected_splice.append(splice_vals[selected_idx])
                    selected_unsplice.append(unsplice_vals[selected_idx])
                    selected_vs.append(pred_vs_vals[selected_idx])
                    selected_vu.append(pred_vu_vals[selected_idx])

        # Handle case where no grid cells meet the density threshold
        if len(selected_splice) == 0:
            print("No grid cells with density above the percentile. Skipping velocity plotting.")
            return

        # Convert lists to arrays
        selected_splice = np.array(selected_splice)
        selected_unsplice = np.array(selected_unsplice)
        selected_vs = np.array(selected_vs)
        selected_vu = np.array(selected_vu)

        # Smooth the velocity components using Gaussian filter
        selected_vs = gaussian_filter(selected_vs, sigma=sigma)
        selected_vu = gaussian_filter(selected_vu, sigma=sigma)

        # Normalize and scale velocity vectors
        magnitude = np.hypot(selected_vs, selected_vu)
        magnitude[magnitude == 0] = 1  # Avoid division by zero
        # # magnitude = 1
        # selected_vs_scaled = (selected_vs / magnitude) * arrow_scale
        # selected_vu_scaled = (selected_vu / magnitude) * arrow_scale
        # Optional: Cap the maximum arrow length
        if max_arrow_length is not None:
            max_magnitude = np.percentile(magnitude, max_arrow_length)
            scale_factors = np.minimum(magnitude, max_magnitude) / magnitude
            selected_vs_scaled = selected_vs * scale_factors
            selected_vu_scaled = selected_vu * scale_factors

        # Downsample the arrows
        downsample_indices = np.arange(0, len(selected_splice), arrow_step)
        selected_splice = selected_splice[downsample_indices]
        selected_unsplice = selected_unsplice[downsample_indices]
        selected_vs_scaled = selected_vs_scaled[downsample_indices]
        selected_vu_scaled = selected_vu_scaled[downsample_indices]

        # Plot velocity arrows using quiver
        q = ax.quiver(
            selected_splice,
            selected_unsplice,
            selected_vs_scaled,
            selected_vu_scaled,
            angles='xy',
            scale_units='xy',
            scale=arrow_scale,
            color=arrow_color,
            width=arrow_width  # Increased arrow width
        )

    # Iterate over each gene and plot
    for idx, gene in enumerate(present_genes):
        ax = axs[idx]
        splice_vals = splice[:, idx]
        unsplice_vals = unsplice[:, idx]

        # Scatter plot
        scatter = ax.scatter(
            splice_vals,
            unsplice_vals,
            c=colors,
            cmap='viridis' if cmap_key not in adata.uns else None,
            alpha=alpha,
            s=point_size
        )

        if velocity:
            pred_vs_vals = pred_vs[:, idx]
            pred_vu_vals = pred_vu[:, idx]

            # Plot velocity arrows
            plot_velocity_arrows(
                ax=ax,
                splice_vals=splice_vals,
                unsplice_vals=unsplice_vals,
                pred_vs_vals=pred_vs_vals,
                pred_vu_vals=pred_vu_vals,
                grid_size=grid_size,
                sigma=sigma,
                lowerb=lowerb,
                upperb=upperb,
                agg=agg,
                arrow_scale=arrow_scale,  # Use arrow_scale to adjust arrow lengths
                arrow_step=arrow_step,     # Downsample the arrows
                arrow_color='black',
                arrow_width=0.005  # Increased arrow width
            )

        # Set labels and title
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(f"{gene}")

        # Adjust axes limits to fit the scatter points with padding
        x_min, x_max = splice_vals.min(), splice_vals.max()
        y_min, y_max = unsplice_vals.min(), unsplice_vals.max()
        x_pad = (x_max - x_min) * 0.05  # 5% padding
        y_pad = (y_max - y_min) * 0.05  # 5% padding
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Adjust layout
    plt.tight_layout()

    # Add legend
    if legend_elements:
        fig.legend(
            handles=legend_elements,
            title=color_by,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.05),
            ncol=len(legend_elements),
            frameon=False
        )

    # Hide any unused subplots
    for j in range(len(present_genes), len(axs)):
        fig.delaxes(axs[j])

    # Save and close the figure
    plt.savefig(path)
    # plt.show()
    plt.close()
