# --- STEER Slide-seq Preprocessing Configuration ---

# Directory settings
BASE_DIR="/nvme/users/liuzhy/TopoVelo_Data/Slideseq_brain_RawData"
ENV_NAME="slideseq_bam"
SAMPLE="Puck_190921_19"

# Input Files
BAM="${BASE_DIR}/${SAMPLE}.bam"
BEAD_CSV="${BASE_DIR}/${SAMPLE}_bead_locations.csv"
GTF="${BASE_DIR}/Mus_musculus.GRCm38.81.gtf"

# Mapping rule confirmed by 00_detect_xc_mapping.py
DROP_IDX_0BASED=7   # Set the optimal drop index here
RC="false"          # "true" or "false"

# Output settings
OUT_DIR="${BASE_DIR}/velocyto_out/${SAMPLE}"
LOG_DIR="${BASE_DIR}/logs"