# Quick Start

## Purpose

This page is the main entry point for users who want to run the core STEER workflow on prepared data.

The recommended first notebook is:

- [Quick Start Notebook](notebooks/quickstart.ipynb)

This notebook is designed for first-time users and presents the STEER workflow in a more structured, guided format.

## Before you begin

STEER can be run directly when your input `.h5ad` file already contains:

- `spliced`
- `unspliced`
- `X_spatial` for the spatial workflow used in the quick start notebook

If these layers are not yet available, please start from the raw-data preprocessing tutorials instead.

## Recommended starting path

For most users, the recommended order is:

1. Complete the installation
2. Open the [Quick Start Notebook](notebooks/quickstart.ipynb)
3. Prepare or load an input `.h5ad`
4. Run the core STEER workflow
5. Explore downstream velocity-related analyses

## Recommended resources

The following repository resources are most relevant to the quick-start workflow:

- [Quick Start Notebook](notebooks/quickstart.ipynb)
- Demo data directory: `tutorials/demo_data/`

Repository links:

- [Open quick start notebook on GitHub](https://github.com/lzygenomics/STEER/blob/master/docs/notebooks/quickstart.ipynb)
- [Open demo data directory on GitHub](https://github.com/lzygenomics/STEER/tree/master/tutorials/demo_data)

## What this quick start is for

The quick start notebook is the best entry point if you want to:

- understand the standard STEER workflow
- run the model on prepared data
- inspect expected inputs and outputs
- understand the role of the major training stages and key parameters before moving on to more advanced analyses

This quick start notebook is the main tutorial entry point for new users.

## Open the notebook

You can access the notebook directly in the documentation:

- [Open Quick Start Notebook](notebooks/quickstart.ipynb)

## Next steps

After completing the quick start, you can continue with:

- [Main Figures](tutorials/main-figures.md) for paper-oriented analyses
- [Raw Data Preprocessing](tutorials/raw-data-preprocessing.md) if your data does not yet include spliced/unspliced layers
- [Citation](citation.md) if you are preparing a manuscript or presentation
