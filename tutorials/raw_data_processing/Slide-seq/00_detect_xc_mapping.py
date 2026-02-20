import argparse
import pysam
import pandas as pd
import collections

def revcomp(s: str) -> str:
    trans = str.maketrans("ACGTN", "TGCAN")
    return s.translate(trans)[::-1]

def drop_pos(s: str, drop_idx: int) -> str:
    return s[:drop_idx] + s[drop_idx+1:]

def main():
    parser = argparse.ArgumentParser(description="Detect optimal alignment strategy for Slide-seq barcodes.")
    parser.add_argument("--bam", required=True, help="Path to the BAM file")
    parser.add_argument("--bead_csv", required=True, help="Path to the bead locations CSV file")
    parser.add_argument("--n_reads", type=int, default=200000, help="Number of reads to scan (default: 200000)")
    args = parser.parse_args()

    print("1) Loading bead barcodes and expanding trailing N -> A/C/G/T ...")
    df = pd.read_csv(args.bead_csv)
    bead14 = df["barcodes"].astype(str).str.upper().tolist()

    bead14_expanded = set()
    for b in bead14:
        if b.endswith("N") and len(b) == 14:
            base = b[:-1]
            for nt in "ACGT":
                bead14_expanded.add(base + nt)
        else:
            bead14_expanded.add(b)

    print(f"   bead14 raw: {len(bead14)} rows")
    print(f"   bead14 expanded unique: {len(bead14_expanded)}")

    print(f"2) Scanning BAM, collecting unique XC=15bp (limit: {args.n_reads}) ...")
    xc15_set = set()
    processed_reads = 0

    with pysam.AlignmentFile(args.bam, "rb") as sam:
        for read in sam.fetch(until_eof=True):
            if processed_reads >= args.n_reads:
                break
            if not read.has_tag("XC"):
                continue
            raw_xc = read.get_tag("XC")
            if raw_xc is None:
                continue
            raw_xc = str(raw_xc).upper()
            if len(raw_xc) != 15:
                continue
            xc15_set.add(raw_xc)
            processed_reads += 1

    print(f"   processed reads (with XC len=15): {processed_reads}")
    print(f"   unique XC15 observed: {len(xc15_set)}")

    print("3) Deletion-position scan (15->14) with rc/no-rc ...")
    hits = collections.defaultdict(int)

    for rc in (False, True):
        for drop_idx in range(15):
            cnt = 0
            for xc in xc15_set:
                x = revcomp(xc) if rc else xc
                b14 = drop_pos(x, drop_idx)
                if b14 in bead14_expanded:
                    cnt += 1
            hits[(rc, drop_idx)] = cnt

    print("\n=== Top 10 strategies by matched unique barcodes ===")
    top = sorted(hits.items(), key=lambda kv: kv[1], reverse=True)[:10]
    for (rc, drop_idx), cnt in top:
        rate = cnt / max(1, len(xc15_set)) * 100
        print(f"rc={rc}\tdrop_idx={drop_idx} (0-based)\tmatches={cnt}\trate={rate:.2f}%")

    best = max(hits.items(), key=lambda kv: kv[1])
    (rc_best, drop_best), cnt_best = best[0], best[1]
    rate_best = cnt_best / max(1, len(xc15_set)) * 100

    print("-" * 60)
    print(f"BEST: rc={rc_best}, drop_idx={drop_best} (0-based), matches={cnt_best}, rate={rate_best:.2f}%")

    if rate_best < 10:
        print("WARNING: Match rate is very low. This suggests the 15->14 rule may not be a single-base deletion, or barcode orientation/format differs.")

if __name__ == "__main__":
    main()