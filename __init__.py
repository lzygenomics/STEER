# STEER/__init__.py

# Training functions
from .training import (
    model_training_share_neighbor_adata,
    model_training_gene_neighbor_adata,
    pretrain_mclust,
    inductive_learn,
    pretrain_mclust_batch,
    inductive_learn_batch,
    model_training_share_neighbor_adata_batch
)

# Prior functions
from .prior import (
    use_scaled_orig,
    assign_random_clusters,
    grid_us_space_extended,
    filter_by_entropy,
    analyze_density_peaks_for_genes,
    pred_regulation,
    assign_random_clusters_to_cells,
    add_annotations_to_df,
    add_entropy_to_adata,
    pred_regulation_anndata
)

# Plot functions
from .plot import (
    embedding_plot,
    Cluster_Results_plot,
    Time_plot,
    plot_regulation,
    heatmap_confusion_gene,
    heatmap_confusion_cell,
    sample_and_plot_us,
    Time_plot_cluster,
    velocity_graph,
    plot_gene_expression,
    coar_adj,
    plot_stacked_bar,
    plot_sankey_from_anndata,
    plot_curve,
    expert_DG,
    expert_DG_multi,
    get_all_significant_genes,
    preprocess_dtw,
    plot_dtw_alignment,
    calculate_lag,
    plot_entropy
)

# Utils functions
from .utils import (
    preprocess,
    preload_datasets_all_genes,
    run_simulations,
    preprocess_anndata,
    results_to_anndata,
    us_moments,
    preload_datasets_all_genes_anndata,
    add_result_anndata,
    preprocess_anndata_fineT,
    add_final_anndata,
    us_transition_matrix,
    add_prior_anndata,
    create_pyg_data,
    fine_preprocess,
    normalize_l2_anndata,
    compute_gene_specific_adj,
    df_2_anndata,
    downsample_adata_by_celltype,
    mclust_R,
    clean_anndata,
    to_dynamo_format,
    downsample_adata_randomly,
    calculate_gene_cosine_similarity,
    cross_boundary_correctness,
    TimeNorm,
    plot_pred_time_correlation,
    calculate_layers_cosine_similarity,
    compare_velocity,
    cross_boundary_correctness_multiple,
    compare_velocity_multiple,
    compare_velocity_umap_other,
    compare_velocity_umap,
    find_pass_markers,
    preprocess_anndata_spatial
)

# Explicitly define all imports for convenience
__all__ = [
    # Training
    'model_training_share_neighbor_adata', 'model_training_gene_neighbor_adata', 'pretrain_mclust',
    'inductive_learn', 'pretrain_mclust_batch', 'inductive_learn_batch', 'model_training_share_neighbor_adata_batch',

    # Prior
    'use_scaled_orig', 'assign_random_clusters', 'grid_us_space_extended', 'filter_by_entropy',
    'analyze_density_peaks_for_genes', 'pred_regulation', 'assign_random_clusters_to_cells',
    'add_annotations_to_df', 'add_entropy_to_adata', 'pred_regulation_anndata',

    # Plot
    'embedding_plot', 'Cluster_Results_plot', 'Time_plot', 'plot_regulation', 'heatmap_confusion_gene',
    'heatmap_confusion_cell', 'sample_and_plot_us', 'Time_plot_cluster', 'velocity_graph',
    'plot_gene_expression', 'coar_adj', 'plot_stacked_bar', 'plot_sankey_from_anndata', 'plot_curve',
    'expert_DG', 'expert_DG_multi', 'get_all_significant_genes', 'preprocess_dtw',
    'plot_dtw_alignment', 'calculate_lag', 'plot_entropy',

    # Utils
    'preprocess', 'preload_datasets_all_genes', 'run_simulations', 'preprocess_anndata',
    'results_to_anndata', 'us_moments', 'preload_datasets_all_genes_anndata', 'add_result_anndata',
    'preprocess_anndata_fineT', 'add_final_anndata', 'us_transition_matrix', 'add_prior_anndata',
    'create_pyg_data', 'fine_preprocess', 'normalize_l2_anndata', 'compute_gene_specific_adj', 'df_2_anndata',
    'downsample_adata_by_celltype', 'mclust_R', 'clean_anndata', 'to_dynamo_format',
    'downsample_adata_randomly', 'calculate_gene_cosine_similarity', 'cross_boundary_correctness',
    'TimeNorm', 'plot_pred_time_correlation', 'calculate_layers_cosine_similarity',
    'compare_velocity', 'cross_boundary_correctness_multiple', 'compare_velocity_multiple',
    'compare_velocity_umap_other', 'compare_velocity_umap', 'find_pass_markers', 'preprocess_anndata_spatial'
]

print("STEER package is imported")