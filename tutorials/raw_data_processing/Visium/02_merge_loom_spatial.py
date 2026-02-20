import argparse
import os
import pandas as pd
import scanpy as sc

def clean_loom_barcodes(obs_names):
    """Clean Velocyto formatting: s1:AAAC...x -> AAAC..."""
    s = pd.Series(obs_names).astype(str)
    s = s.str.replace(r"^.*:", "", regex=True)
    s = s.str.replace(r"x$", "", regex=True)
    return s

def clean_seurat_barcodes(series):
    """Clean Seurat/10x formatting: AAAC...-1 -> AAAC..."""
    s = series.astype(str)
    s = s.str.replace(r"-\d+$", "", regex=True)
    return s

def main():
    parser = argparse.ArgumentParser(description="Merge Velocyto loom with Seurat spatial coordinates.")
    parser.add_argument("--loom", required=True, help="Input .loom file from velocyto")
    parser.add_argument("--pos", required=True, help="Input positions CSV from Seurat")
    parser.add_argument("--meta", required=False, help="Input metadata CSV from Seurat (optional)")
    parser.add_argument("--out", required=True, help="Output .h5ad file path")
    args = parser.parse_args()

    print(f"Loading loom: {args.loom}")
    adata = sc.read_loom(args.loom, validate=False)
    adata.var_names_make_unique()
    
    # Generate clean join keys
    adata.obs["join_key"] = clean_loom_barcodes(adata.obs_names).values

    print(f"Loading spatial positions: {args.pos}")
    pos_df = pd.read_csv(args.pos)
    pos_df["join_key"] = clean_seurat_barcodes(pos_df["Barcode"])
    pos_df[["x", "y"]] = pos_df[["x", "y"]].astype("float32")

    # Merge spatial data
    adata.obs = pd.merge(adata.obs, pos_df.drop(columns=["Barcode"]), on="join_key", how="left")

    # Merge metadata if provided
    if args.meta:
        print(f"Loading metadata: {args.meta}")
        meta_df = pd.read_csv(args.meta)
        meta_df["join_key"] = clean_seurat_barcodes(meta_df["Barcode"])
        adata.obs = pd.merge(adata.obs, meta_df.drop(columns=["Barcode"]), on="join_key", how="left", suffixes=("", "_meta"))

    # Set spatial coordinates in obsm
    matched_mask = adata.obs["x"].notna()
    spatial_coords = adata.obs[["x", "y"]].to_numpy()
    adata.obsm["spatial"] = spatial_coords

    # Filter to keep only spots that mapped to spatial coordinates
    adata_final = adata[matched_mask].copy()
    if "join_key" in adata_final.obs.columns:
        del adata_final.obs["join_key"]

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    adata_final.write_h5ad(args.out, compression="gzip")
    
    match_rate = matched_mask.sum() / adata.n_obs if adata.n_obs > 0 else 0
    print(f"Match Rate: {matched_mask.sum()}/{adata.n_obs} ({match_rate:.1%})")
    print(f"Successfully saved STEER-ready object to: {args.out}")

if __name__ == "__main__":
    main()