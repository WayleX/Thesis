#!/bin/bash
set -e

# ============================================================
#  launch_generation.sh
#  Downloads the DeepFake dataset and launches video generation
#  with either the Wan2.2 or HunyuanVideo-1.5 model.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- defaults ----------
MODEL=""
MODEL_SIZE=""            # wan only: 14b or 5b
DATASET_DIR="$SCRIPT_DIR/DeepFakeDataset_v1.0"
OUTPUT_DIR=""            # set per-model below if not overridden
SKIP_DOWNLOAD=false
SKIP_SETUP=false
GDRIVE_FILE_ID="enter"
ZIP_NAME="DeepFakeDataset_v1.0.zip"

# ---------- usage ----------
usage() {
    cat <<EOF
Usage: $(basename "$0") --model <wan|hunyuan> [OPTIONS]

Required:
  --model <wan|hunyuan>       Which generation model to run.

Options:
  --model-size <14b|5b>       Wan model variant (ignored for hunyuan).
                              Default: 14b
  --dataset-dir <path>        Path to (or destination for) the dataset.
                              Default: $DATASET_DIR
  --output-dir  <path>        Where generated videos are saved.
                              Default: <model_dir>/outputs
  --skip-download             Skip gdown download (dataset already present).
  --skip-setup                Skip repo clone / venv / pip install.
  -h, --help                  Show this help message.
EOF
    exit 0
}

# ---------- arg parsing ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2";        shift 2 ;;
        --model-size)   MODEL_SIZE="$2";   shift 2 ;;
        --dataset-dir)  DATASET_DIR="$2";  shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        --skip-setup)   SKIP_SETUP=true;   shift ;;
        -h|--help)      usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required (wan or hunyuan)"
    usage
fi

MODEL=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')
if [[ "$MODEL" != "wan" && "$MODEL" != "hunyuan" ]]; then
    echo "ERROR: --model must be 'wan' or 'hunyuan', got '$MODEL'"
    exit 1
fi

# ---------- resolve model directory ----------
if [[ "$MODEL" == "wan" ]]; then
    MODEL_DIR="$SCRIPT_DIR/Wan2"
else
    MODEL_DIR="$SCRIPT_DIR/Hunyuan"
fi

[[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="$MODEL_DIR/outputs"

# Default model size for wan
if [[ "$MODEL" == "wan" && -z "$MODEL_SIZE" ]]; then
    MODEL_SIZE="14b"
fi

echo "============================================"
echo "  Model        : $MODEL"
[[ "$MODEL" == "wan" ]] && echo "  Model size   : $MODEL_SIZE"
echo "  Dataset dir  : $DATASET_DIR"
echo "  Output dir   : $OUTPUT_DIR"
echo "  Script dir   : $SCRIPT_DIR"
echo "  Model dir    : $MODEL_DIR"
echo "  Skip download: $SKIP_DOWNLOAD"
echo "  Skip setup   : $SKIP_SETUP"
echo "============================================"

# ==========================================================
#  1. Download & unzip dataset
# ==========================================================
if [[ "$SKIP_DOWNLOAD" == false ]]; then
    # install gdown if missing
    if ! command -v gdown &>/dev/null; then
        echo "[1/3] Installing gdown..."
        pip install --quiet gdown
    fi

    if [[ -d "$DATASET_DIR" && -f "$DATASET_DIR/prompts_v03.csv" ]]; then
        echo "[1/3] Dataset already present at $DATASET_DIR — skipping download."
    else
        echo "[1/3] Downloading dataset from Google Drive..."
        DOWNLOAD_DEST="$SCRIPT_DIR/$ZIP_NAME"
        gdown "https://drive.google.com/uc?id=$GDRIVE_FILE_ID" -O "$DOWNLOAD_DEST"

        echo "[1/3] Unzipping dataset..."
        unzip -q -o "$DOWNLOAD_DEST" -d "$SCRIPT_DIR"
        rm -f "$DOWNLOAD_DEST"
        echo "[1/3] Dataset ready at $DATASET_DIR"
    fi
else
    echo "[1/3] Skipping download (--skip-download)."
    if [[ ! -d "$DATASET_DIR" ]]; then
        echo "WARNING: Dataset directory does not exist: $DATASET_DIR"
    fi
fi

# ==========================================================
#  2. Setup model environment (clone, venv, deps)
# ==========================================================
if [[ "$SKIP_SETUP" == false ]]; then
    echo "[2/3] Setting up $MODEL environment..."
    cd "$MODEL_DIR"

    if [[ "$MODEL" == "wan" ]]; then
        REPO_URL="https://github.com/Wan-Video/Wan2.2"
        REPO_DIR="Wan2.2"
    else
        REPO_URL="https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5"
        REPO_DIR="HunyuanVideo-1.5"
    fi

    VENV_DIR="$REPO_DIR/venv"

    # Clone
    if [[ ! -d "$REPO_DIR" ]]; then
        echo "  Cloning $REPO_URL ..."
        git clone "$REPO_URL"
    else
        echo "  Repo $REPO_DIR already cloned."
    fi

    # Venv
    if [[ ! -d "$VENV_DIR" ]]; then
        echo "  Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"

    # Install deps
    if [[ "$MODEL" == "wan" ]]; then
        if [[ -f "$MODEL_DIR/requirements_wan.txt" ]]; then
            pip install --quiet -r "$MODEL_DIR/requirements_wan.txt"
            pip install --quiet flash-attn==2.7.4.post1 --no-build-isolation
            pip install --quiet pandas
        fi
    else
        if [[ -f "$REPO_DIR/requirements.txt" ]]; then
            pip install --quiet -r "$REPO_DIR/requirements.txt"
            pip install --quiet diffusers==0.36.0
            pip install --quiet flash-attn==2.7.4.post1 --no-build-isolation
            pip install --quiet pandas
        fi

        # Patch diffusers transformer file for HunyuanVideo 1.5
        LOCAL_PATCH="$MODEL_DIR/transfomers_hunyan_video15.py"
        PY_VER=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        TARGET_FILE="$VENV_DIR/lib/python$PY_VER/site-packages/diffusers/models/transformers/transformer_hunyuan_video15.py"
        if [[ -f "$LOCAL_PATCH" && -f "$TARGET_FILE" ]]; then
            cp "$TARGET_FILE" "${TARGET_FILE}.bak_$(date +%Y%m%d_%H%M%S)"
            cp "$LOCAL_PATCH" "$TARGET_FILE"
            echo "  Patched diffusers transformer file."
        fi
    fi

    echo "[2/3] Environment ready."
else
    echo "[2/3] Skipping setup (--skip-setup)."
    # Still activate venv for the python run
    cd "$MODEL_DIR"
    if [[ "$MODEL" == "wan" ]]; then
        VENV_DIR="Wan2.2/venv"
    else
        VENV_DIR="HunyuanVideo-1.5/venv"
    fi
    if [[ -d "$VENV_DIR" ]]; then
        source "$VENV_DIR/bin/activate"
    else
        echo "WARNING: venv not found at $MODEL_DIR/$VENV_DIR — using system python"
    fi
fi

# ==========================================================
#  3. Run generation
# ==========================================================
echo "[3/3] Launching $MODEL video generation..."

if [[ "$MODEL" == "wan" ]]; then
    RUN_SCRIPT="$MODEL_DIR/run_wan.py"
else
    RUN_SCRIPT="$MODEL_DIR/run_hunyuan.py"
fi

EXTRA_ARGS=""
if [[ "$MODEL" == "wan" ]]; then
    EXTRA_ARGS="--model-size $MODEL_SIZE"
fi

python "$RUN_SCRIPT" \
    --dataset-dir "$DATASET_DIR" \
    --output-dir  "$OUTPUT_DIR" \
    $EXTRA_ARGS

echo "Done! Videos saved to $OUTPUT_DIR"
