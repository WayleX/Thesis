#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# --- Configuration ---
SETUP_VENV=true
REPO_URL="https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5"
REPO_DIR="HunyuanVideo-1.5"
VENV_DIR="venv"
START_SCRIPT="/workspace/run_hunyuan.py"

echo "Starting deployment for $REPO_DIR..."

# 1. Clone the repository
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning repository..."
    git clone "$REPO_URL"
    cd "$REPO_DIR"
else
    echo "Directory already exists. Pulling latest changes..."
    cd "$REPO_DIR"
    git pull
fi

if [ "$SETUP_VENV" = true ]; then
    # 2. Set up the virtual environment
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi
fi

if [ "$SETUP_VENV" = true ]; then
    echo "Setting up virtual environment..."
    source "$VENV_DIR/bin/activate"
fi

echo "Installing dependencies..."

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    pip install diffusers==0.36.0
    pip install flash-attn==2.7.4.post1 --no-build-isolation
    pip install pandas
else
    echo "No requirements.txt found. Skipping installation."
fi


# 4. Patch one diffusers file in venv (HunyuanVideo 1.5 transformer)
LOCAL_PATCH_FILE="/workspace/transfomers_hunyan_video15.py"
PY_VER=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
TARGET_FILE="/workspace/$REPO_DIR/$VENV_DIR/lib/python$PY_VER/site-packages/diffusers/models/transformers/transformer_hunyuan_video15.py"

if [ -f "$LOCAL_PATCH_FILE" ] && [ -f "$TARGET_FILE" ]; then
    TS=$(date +%Y%m%d_%H%M%S)
    BACKUP_FILE="${TARGET_FILE}.bak_${TS}"
    cp "$TARGET_FILE" "$BACKUP_FILE"
    cp "$LOCAL_PATCH_FILE" "$TARGET_FILE"
    echo "Patched diffusers file: $TARGET_FILE"
    echo "Backup created: $BACKUP_FILE"
else
    echo "Patch skipped. Missing source or target file."
    echo "   source: $LOCAL_PATCH_FILE"
    echo "   target: $TARGET_FILE"
fi



# 4. Run the script
echo "Running $START_SCRIPT..."
python "$START_SCRIPT"