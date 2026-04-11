# STEER

**STEER** is a graph-attention-based **Spatial-Temporal Explainable Expert** model for **RNA velocity inference**. It is a deep learning framework that leverages spatial-temporal gene expression information and graph attention mechanisms to perform interpretable RNA velocity inference. STEER provides modules for training, visualization, prior construction, and utilities tailored to single-cell and spatial dynamics.

<div class="grid cards" markdown>

- :material-rocket-launch: **Quick Start**

    ---

    Install STEER and run the core workflow on prepared data.

    [Open Quick Start](quickstart.md)

- :material-tools: **Installation**

    ---

    Set up Python, PyTorch, PyG, and optional R dependencies.

    [Open Installation Guide](installation.md)

- :material-book-open-page-variant: **Tutorials**

    ---

    Explore the core pipeline and platform-specific preprocessing workflows.

    [Browse Tutorials](tutorials/index.md)

- :material-format-quote-close: **Citation**

    ---

    Use the published reference and DOI when citing STEER.

    [See Citation](citation.md)

</div>

## Overview

STEER is designed for researchers who want an interpretable framework for RNA velocity inference rather than a purely black-box predictor. By integrating spatial-temporal gene expression signals with graph attention mechanisms, STEER aims to model cell-state dynamics in a way that is both expressive and biologically informative.

The framework supports multiple stages of analysis, including model training, visualization, prior construction, and downstream utilities for single-cell and spatial transcriptomics studies.

## Key Features

- **Interpretable RNA velocity inference** based on a Spatial-Temporal Explainable Expert architecture.
- **Graph-attention-based modeling** for capturing structured relationships in complex transcriptomics data.
- **Support for spatial and single-cell dynamics** across different experimental settings.
- **Modular workflow design** covering training, visualization, prior construction, and utility functions.
- **Tutorial-oriented usage** with both end-to-end examples and preprocessing pipelines for major spatial platforms.

## Documentation Guide

This documentation is organized around the typical STEER workflow:

1. **Installation**  
   Set up the required Python environment, PyTorch, PyG libraries, and optional R dependencies such as `mclust`.

2. **Quick Start**  
   Run the main STEER workflow when your input data already contains `spliced` and `unspliced` layers.

3. **Tutorials**  
   Follow the end-to-end STEER pipeline or choose a platform-specific preprocessing route if spliced/unspliced matrices still need to be generated from raw data.

4. **Citation**  
   Copy the published reference and DOI for manuscripts, presentations, or supplementary materials.

## Tutorials and Preprocessing Workflows

STEER currently provides a core model tutorial together with preprocessing guidance for several major spatial transcriptomics platforms.

Available routes include:

- **STEER Core Pipeline Demo**
- **Slide-seq Pipeline**
- **10x Visium Pipeline**
- **Stereo-seq Pipeline**

If your input `.h5ad` file already contains `spliced` and `unspliced` layers, you can directly run STEER without additional preprocessing. Otherwise, the platform-specific tutorials can help generate these matrices from raw sequencing data.

## Installation Note

We recommend using a virtual environment such as `conda` with **Python 3.9+**. Some STEER features additionally require **R** and the `mclust` package. The installation page provides suggested environment setups tested during development, including GPU-oriented examples based on PyTorch and PyG.

[Go to Installation](installation.md)

## Source Code

The STEER source code, issue tracker, and repository history are available on GitHub:

[STEER on GitHub](https://github.com/lzygenomics/STEER)

## Contact

If you encounter any issues or have questions, feel free to open an issue on GitHub or contact:

**lzy_math@163.com**