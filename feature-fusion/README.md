# Feature Fusion Framework

## Overview
This module contains the codebase for training and evaluating multi-feature fusion models, primarily aimed at detecting generated videos. 

## Directory Structure
- **`data/`**: Logic for dataset handling, dataloaders, and scripts for extracting specific feature representations (`extract_depth.py`, `extract_cotracker.py`, etc.).
- **`models/`**: PyTorch definitions for the detection models.
  - `branches/`: Individual feature processing networks (`depth.py`, `pe.py`, `cotracker.py`, `paper.py`).
  - `fusion/`: Network layers designed to fuse the individual branches (`concat.py`, `cross-attn.py`, `gated.py`).
- **`configs/`**: YAML configuration files defining 26 distinct experimental setups comparing various branches and fusion techniques.
- **`checkpoints/`**: Default directory for saving model checkpoints during training.
- **`results/`**: Directory where evaluation metrics and aggregated results run by `evaluate.py` will be stored.

## Key Files
- `train_exp.py`: The main PyTorch Lightning training script.
- `evaluate.py`: Script to evaluate trained checkpoints on the validation/test sets, computing ROC-AUC and EER (Equal Error Rate).
- `infer.py`: Use this to run a single video / extracted feature set through a trained model for predictions.
- `analysis.py` / `failure_analysis.py`: Tools for plotting distributions, comparing feature importance, and diagnosing hard-to-classify samples.
- `run_all.sh` / `run_gpu.sh`: Shell scripts to run sequential experiments defined in `configs/`.

---

## Usage

### 1. Install Dependencies
Make sure you have all required dependencies installed for training and feature extraction:
```bash
pip install -r requirements.txt
```
Also you need to make sure that you have correctly set up all the models whose fusion you want to use. See [MODELS_SETUP.md](MODELS_SETUP.md) for the full reproducibility guide.

### Inference
For inference on a single video or a folder of videos. See [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) for the full setup walkthrough:
```bash
python infer.py --config configs/11_depth_pe_gated.yaml --ckpt /path/to/checkpoint.ckpt --input /path/to/sample/video.mp4
```
> **Running inference on new videos?** See [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) for a complete step-by-step walkthrough: environment setup, cloning external repos, downloading checkpoints, which lines to change for a new machine, and example commands for all concat-fusion configs.


### Training pipeline
### 2. Feature Extraction
Before training, you need to extract the corresponding features for each branch from your dataset. Use the scripts in `data/extract/`.
```bash
# Example
python data/extract/extract_depth.py --input /path/to/videos --output /path/to/features/depth
python data/extract/extract_cotracker.py --input /path/to/videos --output /path/to/features/cotracker
```

### 3. Training
To train an individual configuration for a specific experiment (e.g., merging PE and Depth representations with a gated mechanism):
```bash
python train_exp.py --config configs/11_depth_pe_gated.yaml --gpu 0 --epochs 25
```

#### To run all 26 experiments sequentially:
```bash
bash run_all.sh 0  # Where 0 is the target GPU ID
```

### 4. Evaluation
Evaluate a trained model to get ROC Curve metrics (AUC and EER):
```bash
python evaluate.py --config configs/11_depth_pe_gated.yaml --ckpt /path/to/checkpoint.ckpt --gpu 0
```

### 5. Inference
For inference on a single video or a folder of videos. See [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) for the full setup walkthrough:
```bash
python infer.py --config configs/11_depth_pe_gated.yaml --ckpt /path/to/checkpoint.ckpt --input /path/to/sample/video.mp4
```

### 6. Analysis
```bash
python failure_analysis.py --ckpt /path/to/model.ckpt
python analysis.py --results_dir results/
```

### Evaluation artifacts

| Resource | Link |
|----------|------|
| All outputs and metrics | [Download](https://drive.google.com/file/d/1r9EKf1gIAJDJ7-pfz4y6XBXjWK87vQJi/view?usp=sharing) |
| Checkpoints folder | [Download](https://drive.google.com/file/d/1cbe9GdxWMOdfTxdauthKC57KeqESepSo/view?usp=sharing) |
| MIDD evaluation & feature fusion experiments files| [Download](https://drive.google.com/file/d/16rdj1D5tawEE6-8Hi5joYIp6BqWNgMG-/view?usp=drive_link) |