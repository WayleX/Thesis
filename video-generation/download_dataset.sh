#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- defaults ----------
GDRIVE_FILE_ID="id"
ZIP_NAME="DeepFakeDataset_v1.0.zip"
DATASET_DIR="${1:-$SCRIPT_DIR/DeepFakeDataset_v1.0}"

if [[ -d "$DATASET_DIR" && -f "$DATASET_DIR/prompts_v03.csv" ]]; then
    echo "Dataset already present at $DATASET_DIR — nothing to do."
    exit 0
fi

if ! command -v gdown &>/dev/null; then
    echo "gdown not found — installing via pip..."
    pip install gdown
fi

DOWNLOAD_DEST="$SCRIPT_DIR/$ZIP_NAME"
echo "Downloading dataset from Google Drive..."
gdown "https://drive.google.com/uc?id=$GDRIVE_FILE_ID" -O "$DOWNLOAD_DEST"

echo "Download complete: $DOWNLOAD_DEST"
ls -lh "$DOWNLOAD_DEST"

# ---------- extract ----------
echo "Extracting dataset..."
unzip -o "$DOWNLOAD_DEST" -d "$SCRIPT_DIR"
