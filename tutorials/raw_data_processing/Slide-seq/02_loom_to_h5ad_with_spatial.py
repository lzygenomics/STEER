import argparse
import re
import numpy as np
import pandas as pd
import scanpy as sc
import loompy

def revcomp(s: str) -> str:
    trans = str.maketrans("ACGTN", "TGCAN")
    return s.translate(trans)[::-1]

def parse_cellid_to_bc15(cellid: str, sample: str) -> str:
    s = str(cellid)
    prefix = sample + ":"
    if s.startswith(prefix):
        s = s[len(prefix):]
    s = re.sub(r"[^ACGTN]+$", "", s.upper())
    s = re.sub(r"[^ACGTN]", "", s)
    return s

def drop_pos(s: str, idx0: int) -> str:
    return s[:idx0] + s[idx0+1:]

def main():
    parser = argparse.ArgumentParser(description="Convert velocyto loom to STEER h5ad with spatial coordinates.")
    parser.add_argument("--loom", required=True, help="Input loom file from velocyto")
    parser.add_argument("--bead_csv", required=True, help="Input bead locations CSV file")
    parser.add_argument("--drop_idx", type=int, required=True, help="0-based index of the base to drop")
    parser.add_argument("--rc", type=str, choices=["True", "False"], default="False", help="Apply reverse complement (True/False)")
    parser.add_argument("--out", required=True, help="Output h5ad file path")
    args = parser.parse_args()

    rc_flag = args.rc == "True"
    sample_name = args.loom.split("/")[-1].replace(".loom", "")

    print(f"Loading loom: {args.loom}")
    adata = sc.read_loom(args.loom, sparse=True)
    adata.var_names_make_unique()

    with loompy.connect(args.loom, validate=False) as ds:
        cellid = ds.ca["CellID"]

    bc15 = [parse_cellid_to_bc15(x, sample_name) for x in cellid]
    adata.obs["barcode15"] = bc15
    adata.obs["barcode15_len"] = adata.obs["barcode15"].map(len)

    adata = adata[adata.obs["barcode15_len"] == 15].copy()
    
    # Apply RC if necessary, then drop position to get 14bp barcode
    if rc_flag:
        adata.obs["barcode14"] = adata.obs["barcode15"].map(lambda s: drop_pos(revcomp(s), args.drop_idx))
    else:
        adata.obs["barcode14"] = adata.obs["barcode15"].map(lambda s: drop_pos(s, args.drop_idx))

    print(f"adata after keeping 15bp: {adata.n_obs} obs, {adata.n_vars} vars")

    print(f"Loading bead locations: {args.bead_csv}")
    bead = pd.read_csv(args.bead_csv)
    bead["barcode14_raw"] = bead["barcodes"].astype(str).str.upper()

    bead_noN = bead[~bead["barcode14_raw"].str.endswith("N")].copy()
    bead_withN = bead[bead["barcode14_raw"].str.endswith("N")].copy()

    expanded = [bead_noN]
    if len(bead_withN) > 0:
        base = bead_withN["barcode14_raw"].str.slice(0, 13)
        for nt in ["A", "C", "G", "T"]:
            tmp = bead_withN.copy()
            tmp["barcode14_raw"] = base + nt
            expanded.append(tmp)

    bead_expanded = pd.concat(expanded, ignore_index=True)
    bead_expanded = bead_expanded.drop_duplicates(subset=["barcode14_raw"]).set_index("barcode14_raw")

    adata.obs["xcoord"] = bead_expanded.reindex(adata.obs["barcode14"])["xcoord"].to_numpy()
    adata.obs["ycoord"] = bead_expanded.reindex(adata.obs["barcode14"])["ycoord"].to_numpy()

    n_total = adata.n_obs
    n_mapped = int(adata.obs["xcoord"].notna().sum())
    print(f"mapped coords: {n_mapped}/{n_total} ({n_mapped/n_total:.1%})")

    adata = adata[adata.obs["xcoord"].notna()].copy()
    print(f"adata after spatial filter: {adata.n_obs} obs, {adata.n_vars} vars")

    # Save to h5ad
    adata.write(args.out)
    print(f"Successfully wrote spatial h5ad object to: {args.out}")

if __name__ == "__main__":
    main()