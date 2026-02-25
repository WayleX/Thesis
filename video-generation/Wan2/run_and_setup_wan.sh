#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# --- Configuration ---
SETUP_VENV=true
REPO_URL="https://github.com/Wan-Video/Wan2.2"
REPO_DIR="Wan2.2"
VENV_DIR="venv"
START_SCRIPT="/workspace/run_wan.py"

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
    echo "🔄 Setting up virtual environment..."
    source "$VENV_DIR/bin/activate"
fi

echo "Installing dependencies..."

if [ -f "../requirements_wan.txt" ]; then
    pip install -r ../requirements_wan.txt
    pip install flash-attn==2.7.4.post1 --no-build-isolation
    pip install pandas
else
    echo "No requirements.txt found. Skipping installation."
fi


# 4. Run the script
echo "Running $START_SCRIPT..."
python "$START_SCRIPT"