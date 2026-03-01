#!/bin/bash
set -euo pipefail

# sett
# ============================================================
#  Hunyuan/setup.sh
#  Clones the HunyuanVideo 1.5 repo, creates a virtual environment,
#  and installs the necessary dependencies.
# ============================================================


# Setting up paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5"
REPO_DIR="$SCRIPT_DIR/HunyuanVideo-1.5"
VENV_DIR="$REPO_DIR/venv"

echo "============================================"
echo "  HunyuanVideo 1.5 — environment setup"
echo "  Script dir : $SCRIPT_DIR"
echo "  Repo dir   : $REPO_DIR"
echo "  Venv dir   : $VENV_DIR"
echo "============================================"

# ---------- clone ----------
if [[ ! -d "$REPO_DIR" ]]; then
    echo "Cloning $REPO_URL ..."
    git clone "$REPO_URL" "$REPO_DIR"
else
    echo "Repo already cloned at $REPO_DIR"
fi

# ---------- venv ----------
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
else
    echo "Venv already exists at $VENV_DIR"
fi

echo "Activating venv..."
source "$VENV_DIR/bin/activate"
echo "Python: $(which python) — $(python --version)"

# ---------- deps ----------
REQ_FILE="$REPO_DIR/requirements.txt"
if [[ -f "$REQ_FILE" ]]; then
    echo "Installing requirements from $REQ_FILE ..."
    pip install -r "$REQ_FILE"
else
    echo "WARNING: $REQ_FILE not found — skipping base requirements"
fi


#other versions of diffusers may not work correctly
echo "Installing diffusers==0.36.0 ..."
pip install diffusers==0.36.0

# this flash-attn is the one that works correctly for me
echo "Installing flash-attn (this may take a while)..."
pip install flash-attn==2.7.4.post1 --no-build-isolation

#just for csv easier to read
echo "Installing pandas..."
pip install pandas

# ---------- patch diffusers transformer as current implementation doesnt correctly utilize flash-attn ----------
# finding file in venv
LOCAL_PATCH="$SCRIPT_DIR/transfomers_hunyan_video15.py"
PY_VER=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
TARGET_FILE="$VENV_DIR/lib/python$PY_VER/site-packages/diffusers/models/transformers/transformer_hunyuan_video15.py"

echo ""
echo "Patching diffusers transformer file..."
echo "  Patch source : $LOCAL_PATCH"
echo "  Target file  : $TARGET_FILE"

# copying patch to diffusers transformer file, or warning if not found
if [[ -f "$LOCAL_PATCH" && -f "$TARGET_FILE" ]]; then
    BACKUP="${TARGET_FILE}.bak_$(date +%Y%m%d_%H%M%S)"
    cp "$TARGET_FILE" "$BACKUP"
    echo "  Backup created: $BACKUP"
    cp "$LOCAL_PATCH" "$TARGET_FILE"
    echo "  Patch applied successfully."
elif [[ ! -f "$LOCAL_PATCH" ]]; then
    echo "  WARNING: Patch source not found — skipping."
elif [[ ! -f "$TARGET_FILE" ]]; then
    echo "  WARNING: Target file not found — skipping."
    echo "  (diffusers may not be installed correctly)"
fi

echo ""
echo "HunyuanVideo 1.5 setup complete."
echo "  Venv: source $VENV_DIR/bin/activate"
