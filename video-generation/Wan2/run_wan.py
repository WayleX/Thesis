import os
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import pandas as pd
from diffusers import WanPipeline, WanImageToVideoPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video, load_image
from PIL import Image

# ── CLI args ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Wan2.2 video generation (I2V or T2V)")
parser.add_argument("--mode", type=str, default="i2v", choices=["i2v", "t2v"],
                    help="Generation mode: i2v (image-to-video) or t2v (text-to-video). Default: i2v")
parser.add_argument("--dataset-dir", type=str, required=True,
                    help="Path to DeepFakeDataset_v1.0")
parser.add_argument("--output-dir", type=str, required=True,
                    help="Directory for generated videos")
parser.add_argument("--model-size", type=str, default="14b", choices=["14b", "5b"],
                    help="Model variant: 14b (720p) or 5b (480p). Default: 14b")
parser.add_argument("--shard-index", type=int, default=0,
                    help="Zero-based shard index for multi-worker runs. Default: 0")
parser.add_argument("--shard-count", type=int, default=1,
                    help="Total number of shards/workers. Default: 1")
args = parser.parse_args()

if args.shard_count < 1:
    raise ValueError("--shard-count must be >= 1")
if args.shard_index < 0 or args.shard_index >= args.shard_count:
    raise ValueError("--shard-index must be in [0, --shard-count)")

DATASET_PATH = args.dataset_dir
OUTPUT_DIR = args.output_dir
MODE = args.mode
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Model config per size and mode ────────────────────────
I2V_CONFIG = {
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

T2V_CONFIG = {
    "14b": {
        "model_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "height": 720,
        "width": 1280,
    },
    "5b": {
        "model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "height": 704,
        "width": 1280,
    },
}

cfg = T2V_CONFIG[args.model_size] if MODE == "t2v" else I2V_CONFIG[args.model_size]

# ── Progress helpers ──────────────────────────────────────
# Single shared file regardless of shard count so you can stop and restart
# with a different number of GPUs without losing progress.
PROGRESS_FILE = os.path.join(OUTPUT_DIR, f".progress_wan_{MODE}_{args.model_size}.txt")


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
print(f"Loading model: {model_id}  (mode={MODE})")

vae = AutoencoderKLWan.from_pretrained(
    model_id, subfolder="vae", torch_dtype=torch.float32
)
vae.enable_tiling()

load_kwargs = dict(
    pretrained_model_name_or_path=model_id,
    vae=vae,
    torch_dtype=torch.bfloat16,
)

if MODE == "t2v":
    pipe = WanPipeline.from_pretrained(**load_kwargs)
else:
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

already_done = 0
shard_total = 0
for _, row in df.iterrows():
    csv_idx = int(_)
    if csv_idx % args.shard_count != args.shard_index:
        continue
    shard_total += 1
    fpath = row.iloc[0]
    rel_out = f"{fpath.replace('.mp4', '')}.mp4"
    out_full = os.path.join(OUTPUT_DIR, rel_out)
    if rel_out in completed or os.path.exists(out_full):
        already_done += 1

mode_label = "T2V" if MODE == "t2v" else "I2V"
print(f"\n{'='*60}")
print(f"  Wan2.2 {mode_label} ({args.model_size.upper()})  |  {NUM_GPUS} GPU(s)  |  bf16  |  {cfg['height']}x{cfg['width']}")
print(f"  Model: {model_id}")
if args.shard_count > 1:
    print(f"  Shard: {args.shard_index + 1}/{args.shard_count}")
print(f"  Global rows: {total}")
print(f"  This shard: {already_done}/{shard_total} already completed")
print(f"  Remaining in shard: {shard_total - already_done}")
print(f"{'='*60}\n")

current = already_done
for index, row in df.iterrows():
    if index % args.shard_count != args.shard_index:
        continue

    file = row.iloc[0]
    prompt = row.iloc[2]

    rel_out = f"{file.replace('.mp4', '')}.mp4"
    output_path = os.path.join(OUTPUT_DIR, rel_out)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if rel_out in completed or os.path.exists(output_path):
        continue

    current += 1
    print(f"[{current}/{shard_total}] Generating {rel_out} ...")

    generator = torch.Generator(device="cpu").manual_seed(42)

    if MODE == "i2v":
        image_path = os.path.join(
            DATASET_PATH, "frames",
            file.replace('.mp4', ''),
            file.split('/')[1].replace('.mp4', '_frame0') + '.jpg'
        )
        image = load_image(image_path)
        image = image.resize((cfg["width"], cfg["height"]), Image.LANCZOS)

        video = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=cfg["height"],
            width=cfg["width"],
            num_frames=81 if args.model_size == "14b" else 24 * 5 + 1,
            guidance_scale=5.0,
            num_inference_steps=50,
            generator=generator,
        ).frames[0]
    else:
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=cfg["height"],
            width=cfg["width"],
            num_frames=81 if args.model_size == "14b" else 24 * 5 + 1,
            guidance_scale=5.0,
            num_inference_steps=50,
            generator=generator,
        ).frames[0]

    export_to_video(video, output_path, fps=16 if args.model_size == "14b" else 24)
    mark_completed(rel_out)
    print(f"  Saved: {output_path}  ({current}/{shard_total})")

print(f"\nShard complete — {current}/{shard_total} videos generated.")
