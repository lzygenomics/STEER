# Quick Start

## Purpose

This page is the main entry point for users who want to run the core STEER workflow.

The primary quick-start resource in the repository is:

- `tutorials/demo_tutorial.ipynb`

This notebook serves as the central STEER pipeline demonstration and is the recommended starting point for most users.

## Before you begin

STEER can be run directly when your input `.h5ad` file already contains:

- `spliced`
- `unspliced`

If these layers are not yet available, please start from the raw-data preprocessing tutorials instead.

## Recommended starting path

For most users, the recommended order is:

1. Complete the installation
2. Open the **Core Pipeline Notebook** page in this documentation site
3. Prepare or load an input `.h5ad`
4. Run the core STEER workflow
5. Explore downstream velocity-related analyses

## Repository resources

The following repository resources are most relevant to the quick-start workflow:

- **Core demo notebook**: `tutorials/demo_tutorial.ipynb`
- **Demo data directory**: `tutorials/demo_data/`

Repository links:

- [Open core demo notebook on GitHub](https://github.com/lzygenomics/STEER/tree/master/tutorials/demo_tutorial.ipynb)
- [Open demo data directory on GitHub](https://github.com/lzygenomics/STEER/tree/master/tutorials/demo_data)

## What this quick start is for

The core demo notebook is the best entry point if you want to:

- understand the standard STEER workflow
- run the model on prepared data
- inspect expected inputs and outputs
- get a practical overview before moving on to figure notebooks or preprocessing pipelines

## Next steps

After completing the quick start, you can continue with:

- **Main Figures** for paper-oriented analyses
- **Raw Data Preprocessing** if your data does not yet include spliced/unspliced layers
- **Citation** if you are preparing a manuscript or presentation
