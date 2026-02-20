#!/bin/bash
#SBATCH -J steer_velo_slideseq
#SBATCH -p large
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

echo "=========================================================="
echo " STEER Preprocessing: Slide-seq Velocyto Pipeline"
echo "=========================================================="

# 1. Load user configuration
if [ ! -f "config.sh" ]; then
    echo "ERROR: config.sh not found. Please copy config.example.sh to config.sh and edit it."
    exit 1
fi
source config.sh

# 2. Setup directories & environments
mkdir -p "${OUT_DIR}" "${LOG_DIR}"
TMP_DIR="${OUT_DIR}/tmp_velocyto_${SAMPLE}_${SLURM_JOB_ID}"
mkdir -p "${TMP_DIR}"

export TMPDIR="${TMP_DIR}"
export TMP="${TMP_DIR}"
export TEMP="${TMP_DIR}"
export LC_ALL=C

# Threads / memory tuning
THREADS="${SLURM_CPUS_PER_TASK:-32}"
SORT_MEM="150G"
VIEW_THREADS=$(( THREADS > 32 ? 32 : THREADS ))

echo "==== Job info ===="
echo "Job started on $(hostname) at $(date)"
echo "SAMPLE=${SAMPLE}"
echo "BAM=${BAM}"
echo "RULE: rc=${RC}, drop_idx_0based=${DROP_IDX_0BASED}"
echo "THREADS=${THREADS}"
echo "TMPDIR=${TMPDIR}"
echo "=================="

# Activate conda env if ENV_NAME is provided
if [ -n "${ENV_NAME:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
fi

echo "samtools: $(samtools --version | head -n 1)"
echo "velocyto: $(velocyto --version 2>/dev/null || true)"
python -c "import sys; print('python:', sys.version.split()[0])"

# Sanity checks
for f in "${BAM}" "${BEAD_CSV}" "${GTF}"; do
  if [ ! -s "${f}" ]; then
    echo "ERROR: missing file: ${f}"
    exit 1
  fi
done

############################################
# Step 0: Index BAM (avoid 'index older than bam')
############################################
echo "[0/5] Index BAM"
samtools index -@ "${VIEW_THREADS}" "${BAM}"

############################################
# Step 1: Build bead.raw.tsv (14bp, may end with N)
############################################
echo "[1/5] Build bead.raw.tsv"
BEAD_RAW="${OUT_DIR}/bead.raw.tsv"
cut -d',' -f1 "${BEAD_CSV}" | tail -n +2 > "${BEAD_RAW}"
echo "bead.raw.tsv lines: $(wc -l < "${BEAD_RAW}")"

############################################
# Step 2: Expand trailing N -> A/C/G/T, sort unique
############################################
echo "[2/5] Build bead.expanded14.sorted.tsv"
BEAD_EXP14_SORTED="${OUT_DIR}/bead.expanded14.sorted.tsv"

awk '{
  if($0 ~ /N$/){
    b=substr($0,1,length($0)-1);
    print b "A"; print b "C"; print b "G"; print b "T";
  } else {
    print $0
  }
}' "${BEAD_RAW}" | \
sort --parallel="${THREADS}" -S "${SORT_MEM}" -T "${TMPDIR}" -u > "${BEAD_EXP14_SORTED}"

echo "bead.expanded14.sorted.tsv lines: $(wc -l < "${BEAD_EXP14_SORTED}")"

############################################
# Step 3: Extract (key14, XC15) from BAM
# Support RC automatically via awk function
############################################
echo "[3/5] Extract XC and build bam.key14_xc15.sorted.tsv"
BAM_KEY14_XC15="${OUT_DIR}/bam.key14_xc15.sorted.tsv"

# Convert RC variable to lowercase for safer comparison
RC_FLAG=$(echo "${RC}" | tr '[:upper:]' '[:lower:]')

samtools view -@ "${VIEW_THREADS}" "${BAM}" | \
awk -v DROP="${DROP_IDX_0BASED}" -v RC_FLAG="${RC_FLAG}" '
BEGIN{
    OFS="\t"
    # mapping for reverse complement
    map["A"]="T"; map["T"]="A"; map["C"]="G"; map["G"]="C"; map["N"]="N";
}
function revcomp(seq,   i, rc_seq, char) {
    rc_seq = ""
    for(i=length(seq); i>=1; i--) {
        char = substr(seq, i, 1)
        rc_seq = rc_seq (char in map ? map[char] : char)
    }
    return rc_seq
}
{
    xc="";
    for(i=12;i<=NF;i++){
        if($i ~ /^XC:Z:/){ xc=$i; sub(/^XC:Z:/,"",xc); break }
    }
    if(xc=="" || length(xc)!=15) next;

    # Determine base sequence to apply the drop rule
    if(RC_FLAG == "true") {
        seq_to_drop = revcomp(xc)
    } else {
        seq_to_drop = xc
    }

    # key14 = delete position DROP (0-based) from 15bp string (awk substr is 1-based)
    key = substr(seq_to_drop, 1, DROP) substr(seq_to_drop, DROP+2);
    
    print key, xc
}' | \
sort --parallel="${THREADS}" -S "${SORT_MEM}" -T "${TMPDIR}" -t $'\t' -k1,1 -u > "${BAM_KEY14_XC15}"

echo "bam.key14_xc15.sorted.tsv lines: $(wc -l < "${BAM_KEY14_XC15}")"

############################################
# Step 4: Join bead expanded14 with BAM key14 -> whitelist of XC15
############################################
echo "[4/5] Join to create exact whitelist15"
WHITELIST15="${OUT_DIR}/${SAMPLE}.barcodes15.whitelist.tsv"

join -t $'\t' -1 1 -2 1 "${BEAD_EXP14_SORTED}" "${BAM_KEY14_XC15}" | \
cut -f2 | \
sort --parallel="${THREADS}" -S 50G -T "${TMPDIR}" -u > "${WHITELIST15}"

echo "Whitelist15 lines: $(wc -l < "${WHITELIST15}")"

############################################
# Step 5: Run velocyto
############################################
echo "[5/5] Run velocyto"
VELO_DIR="${OUT_DIR}/velocyto_run"
mkdir -p "${VELO_DIR}"

velocyto run \
  -o "${VELO_DIR}" \
  -e "${SAMPLE}" \
  -b "${WHITELIST15}" \
  -@ "${VIEW_THREADS}" \
  "${BAM}" \
  "${GTF}"

############################################
# Quick loom sanity check
############################################
if [ -s "${VELO_DIR}/${SAMPLE}.loom" ]; then
  echo "==== Loom layer sums ===="
  python - <<PY
import loompy
p="${VELO_DIR}/${SAMPLE}.loom"
try:
    with loompy.connect(p, validate=False) as ds:
        print("shape:", ds.shape)
        for k in ["spliced","unspliced","ambiguous"]:
            if k in ds.layers:
                print(k, float(ds.layers[k][:,:].sum()))
except Exception as e:
    print("Could not validate loom:", e)
PY
fi

############################################
# Cleanup & Manifest
############################################
MANIFEST="${OUT_DIR}/manifest.txt"
cat > "${MANIFEST}" <<EOF
sample=${SAMPLE}
bam=${BAM}
bead_csv=${BEAD_CSV}
rule_rc=${RC}
rule_drop_idx_0based=${DROP_IDX_0BASED}
slurm_job_id=${SLURM_JOB_ID}
start_time=$(date)
EOF
echo "finish_time=$(date)" >> "${MANIFEST}"

# Clean up large temp directory
rm -rf "${TMP_DIR}"

echo "Job finished at $(date)"