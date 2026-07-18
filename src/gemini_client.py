"""Talks directly to Google's Gemini API (not through OpenRouter). Same job as
or_client.py, same inputs and outputs, so run_chain.py can use either one
interchangeably. Use this to rehearse for free on your Gemini API key before
spending real OpenRouter credits on the report's actual measured runs.

Endpoint and request/response shape confirmed against Google's current docs
(ai.google.dev/api, July 2026): POST to
  https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
with header x-goog-api-key, body contents[].parts[] (text + inline_data), and
generationConfig.responseModalities = ["TEXT", "IMAGE"] to get an image back.
"""
import base64
import csv
import io
import os
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from PIL import Image

import config

load_dotenv(config.PROJECT_ROOT / ".env")

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Gemini's own model id, no "google/" prefix (that prefix is only how
# OpenRouter namespaces it). Pass --model gemini-3.1-flash-image when using
# --provider gemini, or this default is used.
DEFAULT_MODEL = "gemini-3.1-flash-image"


def _api_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to .env as GEMINI_API_KEY=your_key"
        )
    return key


def _log_cost(model, run_name, step, cost, prompt_tokens, completion_tokens,
              latency_s="", note=""):
    """Append one receipt line to the SAME data/costs.csv the OpenRouter
    client writes to, so both providers end up in one ledger."""
    config.COSTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    is_new = not config.COSTS_CSV.exists()
    with open(config.COSTS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["timestamp", "model", "run", "step",
                        "prompt_tokens", "completion_tokens",
                        "cost_usd", "latency_s", "note"])
        w.writerow([datetime.now(timezone.utc).isoformat(), model, run_name,
                    step, prompt_tokens, completion_tokens,
                    f"{cost:.6f}", latency_s, note])


def pil_to_b64(img: Image.Image) -> str:
    """Gemini wants plain base64 (no 'data:image/png;base64,' prefix - that
    prefix is an OpenAI/OpenRouter convention, not a Gemini one)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def edit_image(image, instruction: str, model: str = None,
               run_name: str = "", step: str = "", max_retries: int = 3,
               aspect_ratio: str = None):
    """Apply one edit instruction to one image via the direct Gemini API, OR
    generate a brand new image from text alone if image is None (used by
    generate_image below). aspect_ratio (e.g. "3:4") only makes sense when
    generating from scratch - for edits the model keeps the input's shape.

    Same signature and same return shape as or_client.edit_image:
    (edited PIL image, metadata dict). Drop-in compatible with run_chain.py.
    """
    model = model or DEFAULT_MODEL
    url = API_URL.format(model=model)
    headers = {"x-goog-api-key": _api_key(), "Content-Type": "application/json"}

    parts = [{"text": instruction}]
    if image is not None:
        parts.append({"inline_data": {"mime_type": "image/png",
                                       "data": pil_to_b64(image)}})

    gen_config = {"responseModalities": ["TEXT", "IMAGE"]}
    if aspect_ratio:
        gen_config["imageConfig"] = {"aspectRatio": aspect_ratio}

    payload = {"contents": [{"parts": parts}], "generationConfig": gen_config}

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            r = requests.post(url, json=payload, headers=headers, timeout=300)
            latency_s = round(time.time() - t0, 2)

            if not r.ok:
                try:
                    detail = r.json().get("error", {}).get("message", r.text[:300])
                except Exception:
                    detail = r.text[:300]
                raise RuntimeError(f"Gemini API error {r.status_code}: {detail}")

            data = r.json()
            candidates = data.get("candidates") or []
            if not candidates:
                block_reason = (data.get("promptFeedback") or {}).get(
                    "blockReason", "no candidates returned")
                raise RuntimeError(f"No candidates. Reason: {block_reason}")

            parts = candidates[0].get("content", {}).get("parts", [])
            image_b64, text_reply = None, ""
            for part in parts:
                if "inlineData" in part:
                    image_b64 = part["inlineData"]["data"]
                elif "inline_data" in part:
                    image_b64 = part["inline_data"]["data"]
                elif "text" in part:
                    text_reply += part["text"]

            usage = data.get("usageMetadata", {}) or {}
            meta = {
                "model": f"gemini-direct/{model}",
                "instruction": instruction,
                "prompt_tokens": usage.get("promptTokenCount", ""),
                "completion_tokens": usage.get("candidatesTokenCount", ""),
                "cost_usd": 0.0,  # free tier; update if you move to paid tier
                "latency_s": latency_s,
                "text_reply": text_reply[:500],
                "attempt": attempt,
            }
            _log_cost(meta["model"], run_name, step, 0.0,
                      meta["prompt_tokens"], meta["completion_tokens"],
                      latency_s=latency_s,
                      note="" if image_b64 else "no image returned")

            if not image_b64:
                raise RuntimeError(
                    f"No image returned. Model said: {text_reply[:200]}"
                )

            edited = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
            return edited, meta

        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(3 * attempt)

    raise RuntimeError(f"Edit failed after {max_retries} attempts: {last_err}")


def generate_image(prompt: str, model: str = None, run_name: str = "",
                   step: str = "", max_retries: int = 3,
                   aspect_ratio: str = "3:4"):
    """Text-to-image: create a brand NEW image from a prompt alone, no input
    image. Defaults to a 3:4 vertical portrait, matching our base image
    prompt. Thin wrapper around edit_image(image=None, ...)."""
    return edit_image(None, prompt, model=model, run_name=run_name, step=step,
                      max_retries=max_retries, aspect_ratio=aspect_ratio)


if __name__ == "__main__":
    key_ok = "yes" if os.getenv("GEMINI_API_KEY") else "NO - put it in .env"
    print(f"GEMINI_API_KEY found: {key_ok}")
