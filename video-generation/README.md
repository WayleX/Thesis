# DeepFake Dataset Generation

## 1. Download the Dataset

```bash
bash download_dataset.sh
```

> If the script doesn't work, you can download the dataset manually.

---

## 2. Model Setup

Run the setup scripts to clone repositories and install dependencies into virtual environments.

```bash
# Wan2.2
bash Wan2/setup.sh

# HunyuanVideo 1.5
bash Hunyuan/setup.sh
```

**Compatibility:** Works with CUDA < 13.0 (CUDA 13.0 not fully tested)

### Hunyuan Setup

- Clones the repository and creates a virtual environment
- Installs default requirements with overrides for `flash-attn` and `diffusers`
- Requires swapping a diffusers file for proper `flash-attn` utilization

### Wan Setup

- Clones the repository and creates a virtual environment
- Installs requirements and `flash-attn` for optimal performance

---

## 3. Generate Videos

Both models support two generation modes:

| Mode | Description |
|------|-------------|
| `i2v` (default) | **Image-to-video**: Uses the first frame from the dataset as conditioning |
| `t2v` | **Text-to-video**: Generates purely from text prompt, no image input |

**Model sizes:** Wan2 supports 5B and 14B models (choose one)

### Activate Virtual Environments

```bash
# Wan2.2
source Wan2/Wan2.2/venv/bin/activate

# HunyuanVideo 1.5
source Hunyuan/HunyuanVideo-1.5/venv/bin/activate
```

### Wan2.2 Examples

```bash
# Text-to-video with 14B model (default)
bash Wan2/generate.sh --mode t2v

# Image-to-video with 14B model
bash Wan2/generate.sh --mode i2v

# Text-to-video with 5B model
bash Wan2/generate.sh --mode t2v --model-size 5b

# Image-to-video with 5B model
bash Wan2/generate.sh --mode i2v --model-size 5b

# Generate multiple videos on multiple GPUs
bash Wan2/generate_multiple.sh --mode t2v --model-size 5b --gpus 0,1

# Custom paths (use absolute paths)
bash Wan2/generate.sh --model-size 5b --dataset-dir /data/dataset --output-dir /data/output
```

### HunyuanVideo 1.5 Examples

```bash
# Default generation
bash Hunyuan/generate.sh

# Text-to-video
bash Hunyuan/generate.sh --mode t2v

# Text-to-video with custom output directory
bash Hunyuan/generate.sh --mode t2v --output-dir /data/t2v_output

# Generate multiple videos on multiple GPUs
bash Hunyuan/generate_multiple.sh --mode t2v --gpus 0,1
```
