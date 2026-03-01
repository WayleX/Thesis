#!/bin/bash
set -euo pipefail

# setup
# ============================================================
#  Wan2/setup.sh
#  Clones the Wan2.2 repo, creates a virtual environment,
#  and installs the necessary dependencies.
# ============================================================

# Setting up paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/Wan-Video/Wan2.2"
REPO_DIR="$SCRIPT_DIR/Wan2.2"
VENV_DIR="$REPO_DIR/venv"

echo "============================================"
echo "  Wan2.2 — environment setup"
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
REQ_FILE="$SCRIPT_DIR/requirements_wan.txt"
if [[ -f "$REQ_FILE" ]]; then
    echo "Installing requirements from $REQ_FILE ..."
    pip install -r "$REQ_FILE"
else
    echo "WARNING: $REQ_FILE not found — skipping base requirements"
fi

# this flash-attn is the one that works correctly for me
echo "Installing flash-attn (this may take a while)..."
pip install flash-attn==2.7.4.post1 --no-build-isolation

# just for csv easier to read
echo "Installing pandas..."
pip install pandas

echo ""
echo "Wan2.2 setup complete."
echo "  Venv: source $VENV_DIR/bin/activate"
