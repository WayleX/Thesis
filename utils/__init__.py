"""
Shared utility modules for the dataset processing pipeline.

Modules:
    video           – Video discovery and frame extraction
    io              – JSONL / CSV reading, writing, and resume support
    gemini          – Gemini API client with exponential-backoff retry
    text_detection  – EasyOCR-based text overlay detection
    face_detection  – MTCNN GPU-accelerated face counting
    encoding        – FFmpeg / NVENC video encoding
    curation        – Ethnicity balancing, speaker dedup, stats reports
    analysis        – MediaPipe head-pose, lighting, and motion metrics
"""
