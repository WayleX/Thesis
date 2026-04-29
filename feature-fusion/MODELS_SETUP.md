# Appendix: Reproducible software environment and execution procedure

**Scope.** Only the steps required for a given experiment should be executed. The YAML configuration file (`configs/<name>.yaml`) lists which *branches* are active: `use_paper`, `use_pe`, `use_depth`, and `use_ct`. Each branch corresponds to one row in §4.

---

## 1. Notation: placeholders to substitute

Every placeholder below must be replaced by an **absolute path** or **literal value** appropriate to the reader’s machine. **Do not** leave angle brackets or generic strings in the final commands.

| Placeholder | Meaning | Replace with |
|-------------|---------|----------------|
| `<FINAL_SOLUTION>` | Root directory of this project’s `final_solution` package (contains `train_exp.py`, `infer.py`, `configs/`). | Example: `/home/user/thesis/final_solution` |
| `<DATASETS_ROOT>` | Parent directory of the dataset tree: subdirectories **`ffpp/`**, **`dfdc/`**, and **`deepfake_v1.1/`** must exist beneath it, as consumed by `data/paths.py`. | Example: `/mnt/data/datasets` |
| `<SRC>` | A directory where optional Git clones are stored (arbitrary; must be writable). | Example: `$HOME/src` or `/opt/repos` |
| `<PATH_TO_WEIGHTS_CKPT>` | Filesystem path to the PyTorch checkpoint **`weights.ckpt`** for the DFD-FCG backbone (after downloading it from the [DFD-FCG](https://github.com/aiiu-lab/DFD-FCG) project). | Example: `/downloads/weights.ckpt` before `cp` into `checkpoint/` |
| `<CONFIG_NAME>` | Stem of the configuration file **without** `.yaml`, located under `configs/`. | Example: `03_pe` for `configs/03_pe.yaml` |
| `<GPU_ID>` | Integer CUDA device index. | Typically `0` for a single GPU |
| `<INPUT_VIDEO>` | File or directory of videos for `infer.py`. | Example: `/data/query/video.mp4` |
| `<CHECKPOINT_FILE>` | Trained fusion model checkpoint (Lightning `.ckpt`). | Example: `checkpoints/03_pe/best-epoch=12.ckpt` (relative to `<FINAL_SOLUTION>`) |

---

## 2. Environment variables (mandatory semantics)

These variables are read by **training** (`data/paths.py` for `DATASETS_ROOT`), **feature extraction scripts**, and **`infer.py`**. They should be **exported in the shell** before running the corresponding commands, or persisted in the user’s shell profile (e.g. `~/.bashrc`).

| Variable | When required | Set to |
|----------|----------------|--------|
| `DATASETS_ROOT` | **Always** for training, evaluation, feature extraction, and verification. | Absolute path `<DATASETS_ROOT>` (§1). |
| `DFD_FCG_ROOT` | When `use_paper: true` in the chosen YAML (extraction or inference). | Absolute path to the **root** of the cloned DFD-FCG repository (the directory that contains `src/` after clone). |
| `DFD_FCG_CKPT` | When `use_paper: true` **and** the checkpoint is **not** at `{DFD_FCG_ROOT}/checkpoint/weights.ckpt`. | Absolute path `<PATH_TO_WEIGHTS_CKPT>`. |
| `PERCEPTION_MODELS_ROOT` | When `use_pe: true` (extraction or inference). | Absolute path to the **root** of the cloned `perception_models` repository (the directory that contains the `core` Python package). |

**Default behaviour.** If `DFD_FCG_CKPT` is unset, the paper extractor expects weights at:

`<DFD_FCG_ROOT>/checkpoint/weights.ckpt`

If `PERCEPTION_MODELS_ROOT` is unset, the code falls back to a legacy hard-coded path; **thesis reproduction should always set it explicitly** to avoid machine-specific failures.

**Sanity check** (optional):

```bash
echo "DATASETS_ROOT=${DATASETS_ROOT}"
echo "DFD_FCG_ROOT=${DFD_FCG_ROOT}"
echo "DFD_FCG_CKPT=${DFD_FCG_CKPT}"
echo "PERCEPTION_MODELS_ROOT=${PERCEPTION_MODELS_ROOT}"
```

---

## 3. Prerequisites

- **Hardware:** NVIDIA GPU with CUDA capability sufficient for the installed PyTorch build.
- **Software:** Python 3.10 or newer (or the version fixed by the institution’s conda environment), Git, `pip`, network access for **Hugging Face** and **PyTorch Hub** on first use of those models.
- **Data:** The dataset root `<DATASETS_ROOT>` populated according to the layout implied by `data/paths.py` (face-cropped videos or pre-computed `.pt` features as applicable).

---

## 4. Third-party components by branch

| Branch (YAML flag) | Source of **code** | Source of **weights** |
|--------------------|--------------------|-------------------------|
| Paper (`use_paper`) | Public GitHub: [aiiu-lab/DFD-FCG](https://github.com/aiiu-lab/DFD-FCG) (CVPR 2025; Han *et al.*), cloned to `DFD_FCG_ROOT`. | File `weights.ckpt` at `DFD_FCG_CKPT` or `{DFD_FCG_ROOT}/checkpoint/weights.ckpt` (download from the repository’s published checkpoint, as described in that project’s `readme.md`). |
| PE (`use_pe`) | Public GitHub: [facebookresearch/perception_models](https://github.com/facebookresearch/perception_models), installed at `PERCEPTION_MODELS_ROOT`. | Hugging Face model [`facebook/PE-Core-L14-336`](https://huggingface.co/facebook/PE-Core-L14-336); downloaded automatically on first run with `pretrained=True`, or prefetched with the Hugging Face CLI (§6). |
| Depth (`use_depth`) | No separate Git repository; uses `transformers`. | Hugging Face: `depth-anything/Depth-Anything-V2-Base-hf`; downloaded on first use. |
| CoTracker (`use_ct`) | No separate Git repository; uses `torch.hub`. | Model `cotracker3_offline` from `facebookresearch/co-tracker`; cached under `~/.cache/torch/hub` after first use. |

---

## 5. Step-by-step procedure

### 5.1. Create Python environment and install base requirements

Replace `<FINAL_SOLUTION>` with the absolute path from §1.

```bash
cd <FINAL_SOLUTION>

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

On **Windows**, activation is: `.venv\Scripts\activate`.

---

### 5.2. Install Perception Encoder code (required if `use_pe`)

Replace `<SRC>` with the directory from §1. The clone destination folder name `perception_models` may be changed; if so, **`PERCEPTION_MODELS_ROOT` must point to that folder’s absolute path** (§2).

```bash
mkdir -p <SRC>
cd <SRC>

git clone https://github.com/facebookresearch/perception_models.git
cd perception_models
pip install -e .
```

If the upstream repository specifies compatible PyTorch or CUDA versions, follow **`apps/pe/README.md`** inside `perception_models` before training or extraction.

```bash
cd <FINAL_SOLUTION>
```

---

### 5.3. Install DFD-FCG code and weights (required if `use_paper`)

The Spatial Video Learner code is the public implementation **Towards More General Video-based Deepfake Detection through Facial Component Guided Adaptation for Foundation Model** ([DFD-FCG](https://github.com/aiiu-lab/DFD-FCG), CVPR 2025). Replace `<SRC>` and `<PATH_TO_WEIGHTS_CKPT>` as in §1. Obtain `weights.ckpt` from the checkpoint distribution linked in that repository (see its `readme.md` for the current download location).

```bash
cd <SRC>
git clone https://github.com/aiiu-lab/DFD-FCG.git DFD-FCG
cd DFD-FCG
mkdir -p checkpoint
cp <PATH_TO_WEIGHTS_CKPT> checkpoint/weights.ckpt
```

If the checkpoint remains elsewhere, **omit** the copy and set `DFD_FCG_CKPT` in §5.4 to `<PATH_TO_WEIGHTS_CKPT>`.

```bash
cd <FINAL_SOLUTION>
```

---

### 5.4. Export path variables

Replace every placeholder with absolute paths. Example mapping: if `<SRC>` is `/home/user/src`, then `DFD_FCG_ROOT` may be `/home/user/src/DFD-FCG` and `PERCEPTION_MODELS_ROOT` may be `/home/user/src/perception_models`.

```bash
export DATASETS_ROOT="<DATASETS_ROOT>"

export DFD_FCG_ROOT="<SRC>/DFD-FCG"
export DFD_FCG_CKPT="${DFD_FCG_ROOT}/checkpoint/weights.ckpt"

export PERCEPTION_MODELS_ROOT="<SRC>/perception_models"
```

If `weights.ckpt` is **not** under `checkpoint/weights.ckpt`, set:

```bash
export DFD_FCG_CKPT="<PATH_TO_WEIGHTS_CKPT>"
```

These exports must be active in **every** shell session that runs extraction, training, evaluation, or inference, unless they are appended to `~/.bashrc` (or equivalent).

---

### 5.5. Optional: prefetch Hugging Face weights (offline execution)

Replace `<FINAL_SOLUTION>` and activate the virtual environment as in §5.1.

```bash
cd <FINAL_SOLUTION>
source .venv/bin/activate
pip install huggingface_hub

huggingface-cli download facebook/PE-Core-L14-336
huggingface-cli download depth-anything/Depth-Anything-V2-Base-hf
```

---

### 5.6. Feature extraction (when `.pt` files are not yet present)

Activate the environment, export §5.4 variables, `cd` to `<FINAL_SOLUTION>`, then run **only** the commands for branches enabled in the target YAML.

```bash
cd <FINAL_SOLUTION>
source .venv/bin/activate
# export DATASETS_ROOT, DFD_FCG_*, PERCEPTION_MODELS_ROOT as in §5.4

python data/extract/extract_paper.py --gpu <GPU_ID>
python data/extract/extract_pe.py --gpu <GPU_ID>
python data/extract/extract_depth.py --gpu <GPU_ID>
python data/extract/extract_depth_ffpp.py --dataset ffpp --gpu <GPU_ID>
python data/extract/extract_depth_ffpp.py --dataset dfdc --gpu <GPU_ID>
python data/extract/extract_cotracker.py --gpu <GPU_ID>
```

Parallel sharding, if supported by the script, is controlled by `--shard` and `--num_shards` (see each file’s docstring).

---

### 5.7. Training and evaluation

Replace `<CONFIG_NAME>` and `<GPU_ID>`.

```bash
cd <FINAL_SOLUTION>
source .venv/bin/activate
export DATASETS_ROOT="<DATASETS_ROOT>"

python train_exp.py --config configs/<CONFIG_NAME>.yaml --gpu <GPU_ID>
python evaluate.py --config configs/<CONFIG_NAME>.yaml --gpu <GPU_ID>
```

---

### 5.8. Inference on new videos (optional)

Requires §5.2–5.4 for **each** branch set to `true` in the chosen config.

```bash
cd <FINAL_SOLUTION>
source .venv/bin/activate
# exports from §5.4

python infer.py --input <INPUT_VIDEO> --config configs/<CONFIG_NAME>.yaml --gpu <GPU_ID>
```

For inputs that are **already face-cropped**:

```bash
python infer.py --input <INPUT_VIDEO> --config configs/<CONFIG_NAME>.yaml --gpu <GPU_ID> --cropped
```

To select a specific trained fusion checkpoint:

```bash
python infer.py --input <INPUT_VIDEO> --config configs/<CONFIG_NAME>.yaml --gpu <GPU_ID> \
  --ckpt <CHECKPOINT_FILE>
```

Here `<CHECKPOINT_FILE>` is relative to `<FINAL_SOLUTION>` or an absolute path.

---

### 5.9. Verification against stored benchmark metrics (optional)

This mode recomputes metrics from **saved** `.pt` features and the checkpoint; it does **not** load DFD-FCG, PE, Depth, or CoTracker foundation models.

```bash
cd <FINAL_SOLUTION>
source .venv/bin/activate
export DATASETS_ROOT="<DATASETS_ROOT>"

python infer.py --verify --config configs/<CONFIG_NAME>.yaml --gpu <GPU_ID>
python infer.py --verify-all --gpu <GPU_ID>
```

---

## 6. Summary table: substitution checklist

Before running experiments, confirm:

| Item | Correct when |
|------|----------------|
| `<FINAL_SOLUTION>` | Points to the directory containing `train_exp.py`. |
| `<DATASETS_ROOT>` | Contains `ffpp/`, `dfdc/`, `deepfake_v1.1/` as required by `data/paths.py`. |
| `DFD_FCG_ROOT` | Points to the **top-level** DFD-FCG clone; `weights.ckpt` is at `DFD_FCG_CKPT` or `checkpoint/weights.ckpt` inside it. |
| `PERCEPTION_MODELS_ROOT` | Points to the **top-level** `perception_models` clone after `pip install -e .`. |
| `<CONFIG_NAME>.yaml` | Its `use_*` flags match which extractors were run and which §5.2–5.3 installs were performed. |

