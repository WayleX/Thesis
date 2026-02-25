import os
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import pandas as pd
from diffusers import HunyuanVideo15ImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

# ── CLI args ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="HunyuanVideo 1.5 I2V generation")
parser.add_argument("--dataset-dir", type=str, required=True,
                    help="Path to DeepFakeDataset_v1.0")
parser.add_argument("--output-dir", type=str, required=True,
                    help="Directory for generated videos")
args = parser.parse_args()

DATASET_PATH = args.dataset_dir
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Progress helpers ──────────────────────────────────────
PROGRESS_FILE = os.path.join(OUTPUT_DIR, ".progress_hunyuan.txt")


def load_completed():
    """Return set of output filenames already finished."""
    done = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            for line in f:
                done.add(line.strip())
    return done


def mark_completed(name):
    with open(PROGRESS_FILE, "a") as f:
        f.write(name + "\n")


# ── Model setup ──────────────────────────────────────────

NUM_GPUS = torch.cuda.device_count()

load_kwargs = dict(
    pretrained_model_name_or_path="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v",
    torch_dtype=torch.bfloat16,
)

if NUM_GPUS >= 2:
    load_kwargs["device_map"] = "balanced"

pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(**load_kwargs)

if NUM_GPUS < 2:
    pipe.enable_model_cpu_offload()

pipe.vae.enable_tiling()

# flash_varlen is fast but uses .item() which breaks torch.compile graphs,
# so we pick one or the other — flash_varlen is the better choice here.
pipe.transformer.set_attention_backend("flash_varlen")
# torch.compile skipped: incompatible with flash_varlen (graph breaks) and multi-GPU (no cudagraphs)

# ── Generation loop with progress tracking ────────────────
csv_file = os.path.join(DATASET_PATH, "prompts_v03.csv")
df = pd.read_csv(csv_file)
total = len(df)
completed = load_completed()

# Count already-done (from progress file + existing files)
already_done = 0
for _, row in df.iterrows():
    fpath = row.iloc[0]
    rel_out = f"{fpath.replace('.mp4', '')}.mp4"
    out_full = os.path.join(OUTPUT_DIR, rel_out)
    if rel_out in completed or os.path.exists(out_full):
        already_done += 1

print(f"\n{'='*60}")
print(f"  HunyuanVideo 1.5 I2V  |  {NUM_GPUS} GPU(s)  |  bf16  |  {1+5*24} frames  |  720p")
print(f"  Progress: {already_done}/{total} already completed")
print(f"  Remaining: {total - already_done}")
print(f"{'='*60}\n")

current = already_done
for index, row in df.iterrows():
    file = row.iloc[0]
    prompt = row.iloc[2]

    rel_out = f"{file.replace('.mp4', '')}.mp4"
    output_path = os.path.join(OUTPUT_DIR, rel_out)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if rel_out in completed or os.path.exists(output_path):
        continue

    image_path = os.path.join(
        DATASET_PATH, "frames",
        file.replace('.mp4', ''),
        file.split('/')[1].replace('.mp4', '_frame0') + '.jpg'
    )
    image = load_image(image_path)

    generator = torch.Generator(device="cpu").manual_seed(42)

    current += 1
    print(f"[{current}/{total}] Generating {rel_out} ...")

    video = pipe(
        prompt=prompt,
        image=image,
        generator=generator,
        num_frames=1 + 5 * 24,
        num_inference_steps=20,
    ).frames[0]

    export_to_video(video, output_path, fps=24)
    mark_completed(rel_out)
    print(f"  Saved: {output_path}  ({current}/{total})")

print(f"\nAll done — {current}/{total} videos generated.")

