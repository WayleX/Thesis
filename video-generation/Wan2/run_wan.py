import os
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import pandas as pd
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video, load_image
from PIL import Image

# ── CLI args ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Wan2.2 I2V generation")
parser.add_argument("--dataset-dir", type=str, required=True,
                    help="Path to DeepFakeDataset_v1.0")
parser.add_argument("--output-dir", type=str, required=True,
                    help="Directory for generated videos")
parser.add_argument("--model-size", type=str, default="14b", choices=["14b", "5b"],
                    help="Model variant: 14b (720p) or 5b (480p). Default: 14b")
args = parser.parse_args()

DATASET_PATH = args.dataset_dir
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Model config per size ─────────────────────────────────
MODEL_CONFIG = {
    "14b": {
        "model_id": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "height": 720,
        "width": 1280,
    },
    "5b": {
        "model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "height": 704,
        "width": 1280,
    },
}
cfg = MODEL_CONFIG[args.model_size]

# ── Progress helpers ──────────────────────────────────────
PROGRESS_FILE = os.path.join(OUTPUT_DIR, f".progress_wan_{args.model_size}.txt")


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

model_id = cfg["model_id"]
print(f"Loading model: {model_id}")

vae = AutoencoderKLWan.from_pretrained(
    model_id, subfolder="vae", torch_dtype=torch.float32
)
vae.enable_tiling()

load_kwargs = dict(
    pretrained_model_name_or_path=model_id,
    vae=vae,
    torch_dtype=torch.bfloat16,
)

pipe = WanImageToVideoPipeline.from_pretrained(**load_kwargs)
pipe.enable_model_cpu_offload()

negative_prompt = (
    "overexposed, static, blurry details, subtitles, painting, still image, "
    "worst quality, low quality, JPEG artifacts, ugly, deformed, extra fingers, "
    "poorly drawn hands, poorly drawn face, mutation, mutated limbs, "
    "static frame, cluttered background"
)

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
print(f"  Wan2.2 I2V ({args.model_size.upper()})  |  {NUM_GPUS} GPU(s)  |  bf16  |  81 frames  |  {cfg['height']}x{cfg['width']}")
print(f"  Model: {model_id}")
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
    image = image.resize((cfg["width"], cfg["height"]), Image.LANCZOS)

    generator = torch.Generator(device="cpu").manual_seed(42)

    current += 1
    print(f"[{current}/{total}] Generating {rel_out} ...")

    video = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=cfg["height"],
        width=cfg["width"],
        num_frames=81 if args.model_size == "14b" else 24*5+1,
        guidance_scale=5.0,
        num_inference_steps=50,
        generator=generator,
    ).frames[0]

    export_to_video(video, output_path, fps=16 if args.model_size == "14b" else 24)
    mark_completed(rel_out)
    print(f"  Saved: {output_path}  ({current}/{total})")

print(f"\nAll done — {current}/{total} videos generated.")
