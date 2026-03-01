#!/bin/bash
set -euo pipefail

# ============================================================
#  Hunyuan/generate_multiple.sh
#  Launches multiple HunyuanVideo workers across GPUs.
#  Each worker gets a deterministic shard of the CSV to avoid duplicates.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$SCRIPT_DIR/HunyuanVideo-1.5/venv"
RUN_SCRIPT="$SCRIPT_DIR/run_hunyuan.py"

DATASET_DIR="${PROJECT_ROOT}/DeepFakeDataset_v1.0"
OUTPUT_DIR="$SCRIPT_DIR/outputs"
MODE="i2v"
GPUS=""
MAX_WORKERS=0

usage() {
    cat <<EOF
Usage: $(basename "$0") --gpus <id0,id1,...> [OPTIONS]

Required:
  --gpus <list>            Comma-separated GPU IDs, e.g. 0,1,2,3

Options:
  --mode <i2v|t2v>         Generation mode. Default: i2v
  --dataset-dir <path>     Path to the dataset directory.
                           Default: $DATASET_DIR
  --output-dir <path>      Where generated videos are saved.
                           Default: $OUTPUT_DIR
  --max-workers <n>        Limit workers to first n GPUs in --gpus.
                           Default: all provided GPUs
  -h, --help               Show this help message.
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)         GPUS="$2"; shift 2 ;;
        --mode)         MODE="$2"; shift 2 ;;
        --dataset-dir)  DATASET_DIR="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --max-workers)  MAX_WORKERS="$2"; shift 2 ;;
        -h|--help)      usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "$GPUS" ]]; then
    echo "ERROR: --gpus is required"
    usage
fi

MODE=$(echo "$MODE" | tr '[:upper:]' '[:lower:]')
if [[ "$MODE" != "i2v" && "$MODE" != "t2v" ]]; then
    echo "ERROR: --mode must be 'i2v' or 't2v', got '$MODE'"
    exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: Venv not found at $VENV_DIR"
    echo "       Run Hunyuan/setup.sh first."
    exit 1
fi

if [[ ! -d "$DATASET_DIR" ]]; then
    echo "ERROR: Dataset directory not found: $DATASET_DIR"
    echo "       Run download_dataset.sh first."
    exit 1
fi

if [[ ! -f "$RUN_SCRIPT" ]]; then
    echo "ERROR: Generation script not found: $RUN_SCRIPT"
    exit 1
fi

source "$VENV_DIR/bin/activate"
echo "Python: $(which python) — $(python --version)"

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then
    echo "ERROR: No GPU IDs parsed from --gpus '$GPUS'"
    exit 1
fi

for i in "${!GPU_ARRAY[@]}"; do
    GPU_ARRAY[$i]="${GPU_ARRAY[$i]//[[:space:]]/}"
    if [[ -z "${GPU_ARRAY[$i]}" ]]; then
        echo "ERROR: Empty GPU ID in --gpus list"
        exit 1
    fi
done

WORKER_COUNT=${#GPU_ARRAY[@]}
if [[ "$MAX_WORKERS" -gt 0 ]]; then
    if [[ "$MAX_WORKERS" -lt "$WORKER_COUNT" ]]; then
        WORKER_COUNT=$MAX_WORKERS
    fi
fi

if [[ "$WORKER_COUNT" -lt 1 ]]; then
    echo "ERROR: --max-workers results in zero workers"
    exit 1
fi

LOG_DIR="$OUTPUT_DIR/logs_hunyuan_multi"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "============================================"
echo "  HunyuanVideo 1.5 multi-GPU generation"
echo "  Mode         : $MODE"
echo "  Dataset dir  : $DATASET_DIR"
echo "  Output dir   : $OUTPUT_DIR"
echo "  GPUs         : $GPUS"
echo "  Workers      : $WORKER_COUNT"
echo "  Log dir      : $LOG_DIR"
echo "============================================"

declare -a PIDS=()
declare -a WORKER_GPU=()

for ((shard=0; shard<WORKER_COUNT; shard++)); do
    gpu_id="${GPU_ARRAY[$shard]}"
    log_file="$LOG_DIR/worker_${shard}_gpu${gpu_id}.log"

    echo "Starting worker shard ${shard}/${WORKER_COUNT} on GPU ${gpu_id}"

    CUDA_VISIBLE_DEVICES="$gpu_id" python "$RUN_SCRIPT" \
        --mode "$MODE" \
        --dataset-dir "$DATASET_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --shard-index "$shard" \
        --shard-count "$WORKER_COUNT" \
        > "$log_file" 2>&1 &

    PIDS+=("$!")
    WORKER_GPU+=("$gpu_id")
done

echo "All workers started. Waiting for completion..."

FAIL=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    gpu_id="${WORKER_GPU[$i]}"

    if wait "$pid"; then
        echo "Worker $i (GPU $gpu_id) finished successfully."
    else
        echo "Worker $i (GPU $gpu_id) failed. Check logs in $LOG_DIR"
        FAIL=1
    fi
done

if [[ "$FAIL" -ne 0 ]]; then
    echo "One or more workers failed."
    exit 1
fi

echo "All workers completed successfully. Outputs are in $OUTPUT_DIR"
