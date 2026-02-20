# 10x Visium Preprocessing: Space Ranger → Velocyto → h5ad

This folder provides a complete workflow to generate spatial RNA velocity matrices for 10x Visium data. Unlike Slide-seq, Visium data is typically preprocessed by 10x Genomics **Space Ranger**. 

This tutorial bridges the standard outputs of Space Ranger and Seurat with `velocyto` to produce a STEER-ready `AnnData` (`.h5ad`) object containing exact physical coordinates and spliced/unspliced counts.

---

## Prerequisites

Ensure you have the following software and libraries installed:

**Command-line Tools:**
* `velocyto` (for extracting spliced/unspliced counts from 10x BAM files)

**R Packages:**
* `Seurat`
* `ggplot2`
* `optparse`

**Python Packages:**
* `scanpy`
* `loompy`
* `pandas`

---

## Step 0: Generate `.loom` file from raw BAM (Crucial Step)

If you only have the raw Space Ranger output, you must first generate the `.loom` file containing the spliced/unspliced layers. `velocyto` provides a highly optimized command specifically for 10x Space Ranger outputs.

Run the following command in your terminal:

```bash
velocyto run10x -m repeat_msk.gtf /path/to/spaceranger_output/sample_dir genes.gtf

```

*(Note: `repeat_msk.gtf` is optional but highly recommended to mask expressed repeats. The `genes.gtf` should be the exact same annotation file used by Space Ranger).*

**Output:** A `.loom` file located in `/path/to/spaceranger_output/sample_dir/velocyto/`.

---

## Step 1: Export Spatial Coordinates and Metadata from Seurat

Often, researchers have already processed their Visium data using Seurat (stored as `.Robj` or `.rds`). This step extracts the spatial coordinates (`x`, `y`) and meta-information (e.g., cell types, pathologist annotations) from your Seurat objects.

Run the export script for your sample:

```bash
Rscript 01_export_seurat_spatial.R \
  --input /path/to/sample_1.Robj \
  --outdir exports \
  --sample sample_1

```

**Output:**

* `exports/sample_1_positions_spatial.csv` (Contains Barcode, x, and y)
* `exports/sample_1_metadata.csv` (Contains clustering and other metadata)

---

## Step 2: Merge Loom and Spatial Data into `.h5ad`

A major bottleneck in spatial velocity analysis is the barcode formatting discrepancy between tools:

* **Space Ranger / Seurat:** Usually adds a `-1` suffix (e.g., `AAACAACGAATCGCAC-1`)
* **Velocyto:** Often prepends a sample prefix and appends an `x` (e.g., `sample_1:AAACAACGAATCGCACx`)

This script automatically cleans these barcodes, merges the `.loom` counts with your Seurat spatial coordinates, and filters out unmapped spots to build the final object.

```bash
python 02_merge_loom_spatial.py \
  --loom /path/to/velocyto/sample_1.loom \
  --pos exports/sample_1_positions_spatial.csv \
  --meta exports/sample_1_metadata.csv \
  --out h5ad/sample_1_STEER_ready.h5ad

```

**Output:** A final `.h5ad` file ready for STEER analysis. The object contains:

* `adata.layers["spliced"]`
* `adata.layers["unspliced"]`
* `adata.obsm["spatial"]` (Accurate x/y pixel coordinates)