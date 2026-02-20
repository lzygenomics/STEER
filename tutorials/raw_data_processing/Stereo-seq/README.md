# Stereo-seq Preprocessing: Stereopy → h5ad

Stereo-seq data is unique because its spot identifiers are inherently the physical `x` and `y` coordinates on the chip (formatted as `x_y`). Therefore, unlike Slide-seq or Visium, **you do not need an external CSV file to match spatial coordinates**. 

By leveraging the official [Stereopy RNA Velocity Tutorial](https://stereopy.readthedocs.io/en/latest/Tutorials/RNA_Velocity.html), you can directly generate a `.loom` file and quickly reformat it for STEER.

---

## Step 1: Generate `.loom` file using Stereopy

The `stereopy` package provides the `generate_loom` tool to calculate spliced and unspliced counts from your raw data. Run the following snippet in your Python environment:

```python
from stereo.tools import generate_loom

bgef_file = './sample.tissue.gef' # Your Stereo-seq GEF/BGEF file
gtf_file = './genes.gtf'          # Your reference GTF file
out_dir = './loom_output'         

# Generate the .loom file containing spliced/unspliced matrices
generate_loom(
    gef_path=bgef_file,
    gtf_path=gtf_file,
    bin_type='bins',
    bin_size=100,  # Adjust bin_size according to your downstream analysis!
    out_dir=out_dir
)

```

---

## Step 2: Convert to STEER-Compatible `.h5ad`

The `.loom` file generated in Step 1 will have its cell indices named after their coordinates (e.g., `12450_23400`). We simply need to load this file, parse the indices into `x` and `y` columns, and assign them to the `adata.obsm["spatial"]` matrix required by STEER.

```python
import scanpy as sc

# 1. Load the stereopy-generated loom file
# Note: Update the filename to match the output from Step 1
loom_path = "./loom_output/sample.loom" 
adata = sc.read_loom(loom_path, validate=False)
adata.var_names_make_unique()

# 2. Extract x and y coordinates directly from the index (format: x_y)
adata.obs['x'] = adata.obs_names.map(lambda idx: float(idx.split('_')[0]))
adata.obs['y'] = adata.obs_names.map(lambda idx: float(idx.split('_')[1]))

# 3. Build the spatial matrix for STEER
adata.obsm['spatial'] = adata.obs[['x', 'y']].to_numpy()

# 4. Save the final h5ad object
out_path = "STEER_ready_stereoseq.h5ad"
adata.write_h5ad(out_path, compression="gzip")
print(f"Successfully created STEER input file: {out_path}")
