# Raw Data Preprocessing

This page is for users whose input data does **not** yet contain `spliced` and `unspliced` layers.

If your `.h5ad` file already includes these layers, you can skip preprocessing and go directly to the **Quick Start** or **Core Pipeline Notebook**.

## Purpose

A practical challenge in spatial RNA velocity analysis is generating spliced and unspliced count matrices from raw sequencing data. The STEER repository includes preprocessing resources to support this step before velocity inference.

These resources are organized under:

- `tutorials/raw_data_processing/`

## Available platform routes

The repository currently provides preprocessing routes for major spatial transcriptomics platforms, including:

- **Slide-seq**
- **10x Visium**
- **Stereo-seq**

## Platform notes

### Slide-seq

This route provides an end-to-end workflow for generating spliced and unspliced matrices, including handling barcode mismatch issues and running `velocyto`.

### 10x Visium

This route provides a practical workflow that connects `velocyto`-based processing with Visium and Seurat-style outputs.

### Stereo-seq

This route provides a preprocessing path based on the official `stereopy` ecosystem and Stereo-seq-style data organization.

## Repository location

GitHub directory:

- [raw_data_processing](https://github.com/lzygenomics/STEER/tree/master/tutorials/raw_data_processing)

## Recommended usage path

Choose your starting point as follows:

- already have `spliced/unspliced` layers → go to **Quick Start**
- raw Slide-seq data → use the Slide-seq preprocessing route
- raw 10x Visium data → use the 10x Visium preprocessing route
- raw Stereo-seq data → use the Stereo-seq preprocessing route

## After preprocessing

Once spliced and unspliced matrices are available in your processed input object, continue with:

- **Quick Start**
- **Core Pipeline Notebook**
