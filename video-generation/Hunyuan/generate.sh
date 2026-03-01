#!/bin/bash
set -euo pipefail

# ============================================================
#  Hunyuan/generate.sh
#  Activates the venv and runs HunyuanVideo 1.5 generation.
#  Supports both image-to-video (i2v) and text-to-video (t2v).
#  Assumes setup.sh has already been run.
# ============================================================



# Setting up paths

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$SCRIPT_DIR/HunyuanVideo-1.5/venv"

# ---------- defaults ----------
DATASET_DIR="${PROJECT_ROOT}/DeepFakeDataset_v1.0"
OUTPUT_DIR="$SCRIPT_DIR/outputs"
MODE="i2v"

# ---------- usage ----------
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --mode <i2v|t2v>        Generation mode. Default: i2v
                          i2v = image-to-video (720p model)
                          t2v = text-to-video  (480p model)
  --dataset-dir <path>    Path to the dataset directory.
                          Default: $DATASET_DIR
  --output-dir  <path>    Where generated videos are saved.
                          Default: $OUTPUT_DIR
  -h, --help              Show this help message.
EOF
    exit 0
}

# ---------- arg parsing ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)         MODE="$2";         shift 2 ;;
        --dataset-dir)  DATASET_DIR="$2";  shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        -h|--help)      usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ---------- validate mode ----------
MODE=$(echo "$MODE" | tr '[:upper:]' '[:lower:]')
if [[ "$MODE" != "i2v" && "$MODE" != "t2v" ]]; then
    echo "ERROR: --mode must be 'i2v' or 't2v', got '$MODE'"
    exit 1
fi

echo "============================================"
echo "  HunyuanVideo 1.5 — video generation"
echo "  Mode        : $MODE"
echo "  Dataset dir : $DATASET_DIR"
echo "  Output dir  : $OUTPUT_DIR"
echo "============================================"

# ---------- preflight checks ----------
if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: Venv not found at $VENV_DIR"
    echo "       Run setup.sh first."
    exit 1
fi

if [[ ! -d "$DATASET_DIR" ]]; then
    echo "ERROR: Dataset directory not found: $DATASET_DIR"
    echo "       Run download_dataset.sh first."
    exit 1
fi

# ---------- activate venv ----------
echo "Activating venv..."
source "$VENV_DIR/bin/activate"
echo "Python: $(which python) — $(python --version)"

# ---------- run python script----------
RUN_SCRIPT="$SCRIPT_DIR/run_hunyuan.py"

if [[ ! -f "$RUN_SCRIPT" ]]; then
    echo "ERROR: Generation script not found: $RUN_SCRIPT"
    exit 1
fi

# ---------- launch python script ----------
echo "Launching generation..."
python "$RUN_SCRIPT" \
    --mode        "$MODE" \
    --dataset-dir "$DATASET_DIR" \
    --output-dir  "$OUTPUT_DIR"

echo "Done. Videos saved to $OUTPUT_DIR"
