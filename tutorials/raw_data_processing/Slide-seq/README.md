
# Slide-seq Preprocessing: BAM → Spliced/Unspliced (U/S) → h5ad



This folder provides an end-to-end workflow to generate spliced/unspliced count matrices from raw Slide-seq data. Starting from a BAM file (with `XC` barcode tags), this pipeline resolves barcode discrepancies and exports a STEER-compatible `AnnData` (`.h5ad`) object containing:
- `adata.layers["spliced"]`
- `adata.layers["unspliced"]`
- `adata.obsm["spatial"]` (x/y coordinates)

---

## Prerequisites

Before running this pipeline, ensure you have the following software and Python packages installed in your environment:

**Command-line Tools:**
* `samtools` (for BAM indexing and processing)
* `velocyto` (command-line tool for RNA velocity counting)

**Python Packages:**
* `pysam`
* `pandas`
* `scanpy`
* `loompy`
---


## Inputs Required

1. **BAM file**: Aligned to a reference genome, containing `XC` tags (bead barcode per read).
2. **`bead_locations.csv`**: Must contain the following columns:
   - `barcodes` (Typically 14bp; may end with an ambiguous `N`)
   - `xcoord`
   - `ycoord`
3. **GTF annotation file**: Used for spliced/intronic read assignment by `velocyto`.

---

## Step 0 (Recommended): Detect barcode mapping rule

Slide-seq bead barcodes in the location CSV are typically 14bp, while the BAM `XC` tags are often 15bp. There are usually position shifts (insertions/deletions) and/or reverse-complement encodings between them.

Run the detection script to automatically find the best alignment strategy:

```bash
python 00_detect_xc_mapping.py \
  --bam /path/to/sample.bam \
  --bead_csv /path/to/bead_locations.csv \
  --n_reads 200000

```

*The script will report the optimal `(rc, drop_idx)` setting. For example:*

* `rc = False`
* `drop_idx = 7` *(0-based index, meaning "drop the 8th base of the 15bp BAM barcode")*

---

## Step 1: Build whitelist and run velocyto

`velocyto run` requires a valid barcode whitelist matching the exact BAM tag format (here, 15bp `XC` barcodes). This step automatically:

1. Expands bead barcodes ending with `N` into `{A,C,G,T}`.
2. Matches BAM `XC` tags to bead barcodes under the mapping rule inferred in Step 0.
3. Generates a 15bp exact whitelist for `velocyto`.
4. Runs `velocyto` to produce a `.loom` file containing the spliced/unspliced layers.

First, configure your paths in `config.example.sh` (save it as `config.sh`), then submit the job:

```bash
cp config.example.sh config.sh
# Edit variables in config.sh, then run/submit:
sbatch 01_run_velocyto_from_bam.sbatch

```

**Output:** A velocyto-generated `.loom` file (e.g., `velocyto_out/${SAMPLE}/${SAMPLE}.loom`).

---

## Step 2: Convert loom to h5ad and attach spatial coordinates

The raw `.loom` file lacks spatial information. This step:

* Loads the `.loom` file (retaining spliced/unspliced layers).
* Parses the loom cell IDs back to barcodes.
* Applies the exact same 15bp → 14bp mapping rule used in Step 1.
* Joins the data with `bead_locations.csv` to attach `xcoord` and `ycoord`.
* Filters out unmapped spots and writes a STEER-ready `.h5ad` object.

Run the post-processing script:

```bash
python 02_loom_to_h5ad_with_spatial.py \
  --loom /path/to/sample.loom \
  --bead_csv /path/to/bead_locations.csv \
  --drop_idx 7 \
  --rc False \
  --out /path/to/sample.h5ad

```

**Output:** A final `/path/to/sample.h5ad` file ready for STEER spatial velocity inference.

---

## Notes & Common Pitfalls

* **Missing Tags:** Ensure your BAM file contains `XC:Z:` tags (which store the bead barcodes). If your aligner uses a different tag name (e.g., `CB:Z:`), you will need to adapt the tag name in the scripts.
* **GTF Matching:** Use a GTF file that is strictly consistent with the genome build used for the upstream BAM alignment.
* **Low Mapping Rate:** If Step 0 reports a very low match rate, re-run it with more reads (`--n_reads 1000000`) and verify whether `rc=True` (reverse complement) is required for your specific dataset batch.
* **Ambiguous Bases:** The bead `N` expansion in Step 1 is strictly required for datasets where spatial barcodes end with an `N` in the original location CSV file.