# Dataset Processing Pipeline

End-to-end pipeline for filtering, classifying, curating, and processing a balanced real-video dataset for deepfake detection research.

## Pipeline Overview

```
Source Videos
     |
     v
+--------------------+
| 1. filter_videos   |  Resolution / duration / text overlay / face-count filter
+--------+-----------+  + optional trim + resize (NVENC GPU encode)
         v
+--------------------+
| 2a. classify_videos|  Gemini 2.5 Flash Lite -> class + demographics
+--------+-----------+  
         v
+--------------------+
| 3. curate_dataset  |  Balance ethnicities, cap per class
+--------+-----------+  
         v
+--------------------+
| 4. generate_prompts|  Extract frames + generate prompts
+--------------------+
```

## Project Structure

```
clean_dataset_processing/
|-- README.md                      
|-- requirements.txt               # Python dependencies
|-- config.py                      # Shared constants and thresholds
|
|-- utils/                         # Reusable utility modules
|   |-- __init__.py
|   |-- video.py                   # Video discovery + frame extraction
|   |-- io.py                      # JSONL/CSV I/O + resume support
|   |-- gemini.py                  # Gemini API client with retry logic
|   |-- text_detection.py          # EasyOCR text overlay detection
|   |-- face_detection.py          # MTCNN GPU face counting
|   |-- encoding.py                # FFmpeg / NVENC video encoding
|   |-- curation.py                # Ethnicity balancing, speaker dedup, stats
|
|-- filter_videos.py
|-- classify_videos.py
|-- generate_prompts.py
|-- curate_dataset.py
```


## Step-by-Step Usage

### Prerequisites

```bash
pip install -r requirements.txt
# Also ensure: ffmpeg, ffprobe, and CUDA toolkit are installed

# Set up Gemini API key for classify and generate_prompts steps:
echo "GEMINI_API_KEY=your_key_here" > .env
# Or export as environment variable:
export GEMINI_API_KEY="your_key_here"
```

### Step 1: Filter Videos

Scans source videos and keeps only those matching resolution, duration, and face-count criteria. Rejects videos with text overlays.

```bash
python filter_videos.py \
    --input /path/to/source_videos \
    --output /path/to/filtered_output \
    --trim --trim-duration 8 --resize 1280x720 \
    --report filter_report.csv \
    --probe-workers 16 --encode-workers 4
```

**Three-phase pipeline:**
1. **ffprobe** (threaded) -- resolution + duration checks
2. **EasyOCR + MTCNN** (GPU) -- text overlay + face count checks
3. **ffmpeg** (threaded, NVENC) -- encode passing videos

### Step 2: Classify Videos

Extracts 4 frames per video and sends them to Gemini for classification + demographics

```bash
python classify_videos.py \
    --input /path/to/filtered_output \
    --output classification.csv \
    --output-json classification.jsonl \
    --workers 5 \
```


### Step 3: Curate Balanced Dataset

Selects a balanced subset from classified videos with these rules:
- **N videos per class** (default 100)
- **talkingcelebs**: only in Official Statement
- **Ethnicity balancing**

```bash
python curate_dataset.py \
    --input /path/to/filtered_output \
    --jsonl classification.jsonl \
    --output /path/to/curated_output \
    --per-class 100 \
    --seed 42 \
    --stats-csv curation_stats.csv
```
### Step 4: Generate Prompts

Generates detailed video-generation prompts

```bash
python generate_prompts.py \
    --input /path/to/curated_dataset \
    --output prompts.csv \
    --output-json prompts.jsonl \
    --frames-dir /path/to/curated_dataset/frames \
    --model-context "" \
    --workers 5 \
```


After completing you should get curated dataset with prompts + first frame for each video, ready to be inputted to the video generation model
