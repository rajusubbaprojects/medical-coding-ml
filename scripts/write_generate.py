from pathlib import Path

CONTENT = '''"""Vertex AI Gemini wrapper for note generation.

Single function: generate_note(prompt) -> dict with text + token counts.
Uses google-genai in Vertex AI mode (ADC auth, no API key).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

PROJECT = os.environ.get("GCP_PROJECT", "medical-coding-ml-9848")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_OUTPUT_TOKENS = 1500

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def get_client() -> genai.Client:
    """Lazy singleton. Vertex AI mode uses ADC (gcloud auth)."""
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=PROJECT,
            location=LOCATION,
        )
        logger.info("Initialized Gemini client (project=%s, location=%s, model=%s)",
                    PROJECT, LOCATION, MODEL)
    return _client


# Retry only on transient errors. 4xx errors (permission, bad request) raise immediately.
@retry(
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def generate_note(
    prompt: str,
    *,
    model: str = MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> dict[str, Any]:
    """Call Gemini, return text + metadata.

    Returns:
        {
            "text": str,                # the generated note
            "input_tokens": int,
            "output_tokens": int,
            "model": str,               # e.g. "gemini-2.5-flash"
            "finish_reason": str,
        }
    """
    client = get_client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    usage = response.usage_metadata
    candidate = response.candidates[0] if response.candidates else None
    finish_reason = str(candidate.finish_reason) if candidate else "UNKNOWN"

    return {
        "text": response.text or "",
        "input_tokens": usage.prompt_token_count if usage else 0,
        "output_tokens": usage.candidates_token_count if usage else 0,
        "model": model,
        "finish_reason": finish_reason,
    }


def main() -> None:
    """Smoke test: one short hardcoded prompt, print result + token counts."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    prompt = (
        "Write one sentence describing what a discharge summary is, "
        "in plain English, as if explaining to a patient."
    )
    print(f"Prompt: {prompt}\\n")

    result = generate_note(prompt, max_output_tokens=500)

    print(f"--- RESPONSE ---")
    print(result["text"])
    print(f"\\n--- META ---")
    print(f"model:         {result['model']}")
    print(f"input_tokens:  {result['input_tokens']}")
    print(f"output_tokens: {result['output_tokens']}")
    print(f"finish_reason: {result['finish_reason']}")


if __name__ == "__main__":
    main()
'''

Path("src/notes/generate.py").write_text(CONTENT)
print("Wrote src/notes/generate.py")
print(f"  {len(CONTENT.splitlines())} lines")