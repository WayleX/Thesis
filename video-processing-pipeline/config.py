"""
Shared configuration constants for the dataset processing pipeline.

All pipeline scripts import from here to ensure consistent settings
across filtering, classification, curation, and final processing.
"""

# ── Video file detection ──────────────────────────────────────────────────────
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}

# ── Filtering thresholds ──────────────────────────────────────────────────────
ALLOWED_RESOLUTIONS = {(1280, 720), (1920, 1080)}
MIN_DURATION_SEC = 5.0
REQUIRED_PERSON_COUNT = 1
FACE_SAMPLE_FRAMES = 5
FACE_MIN_DETECTION_CONFIDENCE = 0.9

# ── Text detection (EasyOCR) ──────────────────────────────────────────────────
TEXT_MIN_CHARS_PER_FRAME = 3
TEXT_MIN_FRAMES_WITH_TEXT = 1 

# ── Classification labels ─────────────────────────────────────────────────────
CLASSIFICATION_CATEGORIES = [
    "Official Statement",
    "Studio Interview/Podcast",
    "Direct-to-Camera / Casual",
    "Other",
]

FOLDER_MAP = {
    "Official Statement": "Official_Statement",
    "Studio Interview": "Studio_Interview",
    "Direct-to-Camera / Casual": "Direct_to_Camera_Casual",
    "Other": "Other",
}

ETHNICITY_ORDER = [
    "White",
    "Black",
    "Asian",
    "Latino/Hispanic",
    "Middle Eastern",
    "Indian/South Asian",
    "Other",
]

GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_MAX_RETRIES = 5
GEMINI_INITIAL_BACKOFF = 2   
GEMINI_MAX_BACKOFF = 60

ANALYSIS_WINDOW_FRAMES = 30

