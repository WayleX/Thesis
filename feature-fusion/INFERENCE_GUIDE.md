# Inference Guide: Feature-Fusion Deepfake Detection

End-to-end guide for running inference

---

## 1. Repository layout

```
Thesis/
├── feature-fusion/          
│   ├── .venv/              
│   ├── checkpoints/         
│   ├── configs/             
│   ├── data/
│   │   ├── paths.py         
│   │   └── extract/         
│   ├── models/             
│   ├── infer.py             
│   ├── train_exp.py        
│   └── evaluate.py          
├── checkpoints/             
│   ├── 01_paper/best-epoch=19.ckpt
│   ├── 10_all_concat_s3/best-epoch=18.ckpt
│   └── ...
├── DFD-FCG/               
│   └── checkpoint/weights.ckpt
├── perception_models/               
```

---

## 2. One-time setup

### 2.1 Python virtual environment

```bash
cd feature-fusion
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2.1b `lightning` compatibility shim (required for paper branch)

DFD-FCG internally imports `lightning` (Lightning AI's new unified namespace),
but `requirements.txt` installs `pytorch-lightning` (the old alias). Run this once
after creating the venv to create a shim that maps `lightning.*` transparently:

```bash
cd feature-fusion
SITE=".venv/lib/python3.12/site-packages"
mkdir -p "$SITE/lightning/pytorch" "$SITE/lightning/fabric/utilities"

cat > "$SITE/lightning/__init__.py" << 'EOF'
import sys, pytorch_lightning as _pl, lightning_fabric as _lf
sys.modules.setdefault("lightning.pytorch", _pl)
sys.modules.setdefault("lightning.fabric", _lf)
import lightning_fabric.utilities, lightning_fabric.utilities.types
sys.modules.setdefault("lightning.fabric.utilities", lightning_fabric.utilities)
sys.modules.setdefault("lightning.fabric.utilities.types", lightning_fabric.utilities.types)
EOF

echo "from pytorch_lightning import *" > "$SITE/lightning/pytorch/__init__.py"
echo "from lightning_fabric import *"  > "$SITE/lightning/fabric/__init__.py"
echo "from lightning_fabric.utilities import *" > "$SITE/lightning/fabric/utilities/__init__.py"

cat > "$SITE/lightning/fabric/utilities/types.py" << 'EOF'
from lightning_fabric.utilities.types import *
try:
    from lightning_fabric.utilities.types import _PATH
except ImportError:
    from pathlib import Path; _PATH = Path
EOF
```

### 2.2 Clone external repos

```bash
cd Thesis

# Paper branch (DFD-FCG, CVPR 2025)
git clone https://github.com/aiiu-lab/DFD-FCG.git DFD-FCG

# PE branch (Meta Perception Encoder)
git clone https://github.com/facebookresearch/perception_models.git perception_models
```

### 2.3 Install Perception Encoder

```bash
source .venv/bin/activate
cd perception_models
pip install -e .
```

### 2.4 Download DFD-FCG weights

The checkpoint is hosted on Google Drive. Download ID: `1ydD5rnaaF0i2zLE7NidLtAhjonHoVQOk`

> **Important:** The Google Drive file is a ZIP container (~656 MB) that holds
> `checkpoint/weights.ckpt` (1.2 GB extracted) and `checkpoint/setting.yaml` inside it.
> You must extract it after downloading.

```bash
source feature-fusion/.venv/bin/activate
pip install gdown

mkdir -p DFD-FCG/checkpoint
cd DFD-FCG/checkpoint

gdown 1ydD5rnaaF0i2zLE7NidLtAhjonHoVQOk -O weights_container.zip

# Extract the actual weights (creates checkpoint/weights.ckpt and checkpoint/setting.yaml)
python3 -c "
import zipfile, shutil, os
with zipfile.ZipFile('weights_container.zip') as z:
    z.extractall('/tmp/dfd_extracted/')
shutil.copy('/tmp/dfd_extracted/checkpoint/weights.ckpt', 'weights.ckpt')
shutil.copy('/tmp/dfd_extracted/checkpoint/setting.yaml', 'setting.yaml')
print('Extracted OK:', os.path.getsize('weights.ckpt'), 'bytes')
"
```

### 2.5 Prefetch HuggingFace models (optional, for offline use)

Depth and PE models download automatically on first run. Pre-fetch them:

```bash
source .venv/bin/activate
huggingface-cli download depth-anything/Depth-Anything-V2-Base-hf
huggingface-cli download facebook/PE-Core-L14-336
```

---

## 3. Path constants to update in `infer.py`

**File:** `feature-fusion/infer.py`, **lines 43–45**

```python
# Line 43 — root of the DFD-FCG clone
DFD_FCG_ROOT = Path("path/to/DFD-FCG")

# Line 44 — path to DFD-FCG weights (derived from line 43; change only if weights live elsewhere)
DFD_FCG_CKPT = DFD_FCG_ROOT / "checkpoint" / "weights.ckpt"

# Line 45 — root of the perception_models clone
PE_ROOT = Path("path/to/perception_models")
```

**Template for a new machine** (replace `<BASE>` with your project root):

```python
DFD_FCG_ROOT = Path("<BASE>/DFD-FCG")
DFD_FCG_CKPT = DFD_FCG_ROOT / "checkpoint" / "weights.ckpt"
PE_ROOT = Path("<BASE>/perception_models")
```

> **Note:** `data/paths.py` line 12 (`DATASETS = Path(...)`) is only needed for
> training and offline evaluation — it is not read during `infer.py` inference.

---

## 4. Running inference

### 4.1 Activate the environment

```bash
cd feature-fusion
source .venv/bin/activate
```

### 4.2 Single video, single model

```bash
python infer.py \
  --input /path/to/video.mp4 \
  --config configs/10_all_concat_s3.yaml \
  --gpu 0
```

### 4.3 Two test videos, one model

```bash
python infer.py \
  --input path/test_videos/ \
  --config configs/10_all_concat_s3.yaml \
  --gpu 0 \
  --output results/inference_test/10_all_concat_s3.json
```

### 4.4 Explicit checkpoint path

```bash
python infer.py \
  --input /path/to/video.mp4 \
  --config configs/10_all_concat_s3.yaml \
  --ckpt Thesis/checkpoints/10_all_concat_s3/best-epoch=18.ckpt \
  --gpu 0
```


## 5. Adapting to a different machine

1. **Clone this repo** to any path, call it `<BASE>`.
2. **Clone DFD-FCG** and **perception_models** anywhere (e.g. `<BASE>/DFD-FCG`, `<BASE>/perception_models`).
3. **Edit `infer.py` lines 43–45** to point `DFD_FCG_ROOT` and `PE_ROOT` at your clone locations.
4. **Download DFD-FCG weights** to `<DFD_FCG_ROOT>/checkpoint/weights.ckpt`.
5. **Create venv**, install `requirements.txt`, install `perception_models` with `pip install -e .`.
6. Run inference as in section 4.

The only file that must change is **`infer.py` lines 43–45**. No other source file encodes machine-specific paths for inference.

---

## 6. Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `DFD-FCG repo not found` | `DFD_FCG_ROOT` path wrong | Update `infer.py` line 43 |
| `Perception models repo not found` | `PE_ROOT` path wrong | Update `infer.py` line 45 |
| `No checkpoint found in ...` | Symlink missing | Re-run section 2.6 |
| `No module named 'open_clip'` | Missing extra dep | `pip install open-clip-torch==2.24.0` |
| `No module named 'lightning'` | Missing shim | Run section 2.1b to create the shim |
| `Expected hasRecord("version") ...` | Loaded the ZIP container not the real ckpt | Re-extract `weights.ckpt` per section 2.4 |
| `CUDA out of memory` | GPU too small | Reduce load or pass `--cropped` |
| HF model download hangs | No internet | Pre-fetch with `huggingface-cli download` (section 2.5) |
