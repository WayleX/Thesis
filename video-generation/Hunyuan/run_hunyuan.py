import os
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import pandas as pd
from diffusers import HunyuanVideo15Pipeline, HunyuanVideo15ImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

# ── CLI args ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="HunyuanVideo 1.5 video generation (I2V or T2V)")
parser.add_argument("--mode", type=str, default="i2v", choices=["i2v", "t2v"],
                    help="Generation mode: i2v (image-to-video) or t2v (text-to-video). Default: i2v")
parser.add_argument("--dataset-dir", type=str, required=True,
                    help="Path to DeepFakeDataset_v1.0")
parser.add_argument("--output-dir", type=str, required=True,
                    help="Directory for generated videos")
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

MODEL_IDS = {
    "i2v": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v",
    "t2v": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
}
model_id = MODEL_IDS[MODE]

# ── Progress helpers ──────────────────────────────────────
# Single shared file regardless of shard count so you can stop and restart
# with a different number of GPUs without losing progress.
PROGRESS_FILE = os.path.join(OUTPUT_DIR, f".progress_hunyuan_{MODE}.txt")


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

print(f"Loading model: {model_id}  (mode={MODE})")

load_kwargs = dict(
    pretrained_model_name_or_path=model_id,
    torch_dtype=torch.bfloat16,
)

if NUM_GPUS >= 2:
    load_kwargs["device_map"] = "balanced"

if MODE == "t2v":
    pipe = HunyuanVideo15Pipeline.from_pretrained(**load_kwargs)
else:
    pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(**load_kwargs)

if NUM_GPUS < 2:
    pipe.enable_model_cpu_offload()

pipe.vae.enable_tiling()

# flash_varlen is fast but uses .item() which breaks torch.compile graphs,
# so we pick one or the other — flash_varlen is the better choice here.
pipe.transformer.set_attention_backend("flash_varlen")

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

resolution = "720p" if MODE == "i2v" else "480p"
mode_label = "T2V" if MODE == "t2v" else "I2V"
print(f"\n{'='*60}")
print(f"  HunyuanVideo 1.5 {mode_label}  |  {NUM_GPUS} GPU(s)  |  bf16  |  {1+5*24} frames  |  {resolution}")
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

        video = pipe(
            prompt=prompt,
            image=image,
            generator=generator,
            num_frames=1 + 5 * 24,
            num_inference_steps=40,
        ).frames[0]
    else:
        video = pipe(
            prompt=prompt,
            generator=generator,
            num_frames=1 + 5 * 24,
            num_inference_steps=40,
        ).frames[0]

    export_to_video(video, output_path, fps=24)
    mark_completed(rel_out)
    print(f"  Saved: {output_path}  ({current}/{shard_total})")

print(f"\nShard complete — {current}/{shard_total} videos generated.")
