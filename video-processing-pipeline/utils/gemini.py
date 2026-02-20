"""
Gemini API client with exponential-backoff retry logic.

Wraps google-genai so that every LLM-based step shares the same retry / JSON-parsing / error handling.
"""

import json
import time
import logging

from google import genai
from google.genai import types
from PIL import Image

from .video import pil_to_bytes
from config import (
    GEMINI_MODEL,
    GEMINI_MAX_RETRIES,
    GEMINI_INITIAL_BACKOFF,
    GEMINI_MAX_BACKOFF,
)

log = logging.getLogger(__name__)

RETRYABLE_SIGNALS = ["503", "429", "RESOURCE_EXHAUSTED", "overloaded"]


def create_client(api_key: str) -> genai.Client:
    """Create an authenticated Gemini API client."""
    return genai.Client(api_key=api_key)


def strip_markdown_fences(text: str) -> str:
    """Remove ```json … ``` wrappers that LLMs sometimes emit."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()
    return text


def call_with_retry(
    client: genai.Client,
    frames: list[Image.Image],
    system_prompt: str,
    user_text: str,
    *,
    model: str = GEMINI_MODEL,
    temperature: float = 0.2,
    max_output_tokens: int = 1024,
    fallback: dict | None = None,
) -> dict:
    """Send frames + text to Gemini, parse JSON, retry on transient errors.
    """
    parts: list[types.Part] = []
    for frame in frames:
        img_bytes = pil_to_bytes(frame)
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
    parts.append(types.Part.from_text(text=user_text))

    backoff = GEMINI_INITIAL_BACKOFF
    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=types.Content(role="user", parts=parts),
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )

            if response.text is None:
                return fallback or {"_empty_response": True}

            text = strip_markdown_fences(response.text)
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                log.warning("Failed to parse JSON: %s", text[:200])
                return {"_raw": text, "_parse_error": True}

        except Exception as e:
            err_str = str(e)
            retryable = any(code in err_str for code in RETRYABLE_SIGNALS)
            if retryable and attempt < GEMINI_MAX_RETRIES:
                log.warning(
                    "Attempt %d/%d failed (retryable): %s — backing off %.0fs",
                    attempt, GEMINI_MAX_RETRIES, err_str[:100], backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, GEMINI_MAX_BACKOFF)
                continue
            raise

    return {}
