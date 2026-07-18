"""Talks to OpenRouter. One job: send an image plus an instruction, get the
edited image back.

It also logs what every single call cost and how long it took (OpenRouter
reports the exact USD amount in the response when you ask for it) and refuses
to spend past MAX_BUDGET_USD.
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

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# OpenRouter recently split out a separate, dedicated Image API
# (POST /v1/images) with its own request/response shape. Some models are
# ONLY served through that endpoint and 404 on the older chat-completions
# endpoint above - Seedream 4.5 is one of them. Any model slug listed here
# gets automatically routed to the newer endpoint inside edit_image() below,
# so callers never need to know or care which one a given model requires.
IMAGES_API_URL = "https://openrouter.ai/api/v1/images"
IMAGES_API_MODELS = {"bytedance-seed/seedream-4.5"}


class BudgetExceeded(RuntimeError):
    pass


def _api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env "
            "and paste your key."
        )
    return key


def spent_so_far() -> float:
    """Total USD spent so far, read from the receipts file."""
    if not config.COSTS_CSV.exists():
        return 0.0
    total = 0.0
    with open(config.COSTS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            try:
                total += float(row["cost_usd"])
            except (KeyError, ValueError):
                pass
    return total


def _log_cost(model, run_name, step, cost, prompt_tokens, completion_tokens,
              latency_s="", note=""):
    """Append one receipt line to data/costs.csv."""
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


def pil_to_data_url(img: Image.Image) -> str:
    """Pack a picture into a text string so it can travel inside a web request."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def data_url_to_pil(url: str) -> Image.Image:
    """Unpack the picture the model sends back."""
    b64 = url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _edit_image_via_images_api(image, instruction, model, run_name, step, max_retries):
    """Same job as edit_image, but calls OpenRouter's newer dedicated Image
    API (POST /v1/images) instead of the chat-completions endpoint. Request
    shape is different: {"model", "prompt", "input_references"} instead of
    a messages/content array. Response shape is different too: images come
    back as data[].b64_json instead of choices[0].message.images. Used
    automatically for any model listed in IMAGES_API_MODELS above - callers
    just use edit_image()/generate_image() as normal.
    """
    if spent_so_far() >= config.MAX_BUDGET_USD:
        raise BudgetExceeded(
            f"Spent ${spent_so_far():.2f} already, cap is "
            f"${config.MAX_BUDGET_USD:.2f}. Raise MAX_BUDGET_USD in "
            f"config.py only if you are sure."
        )

    payload = {"model": model, "prompt": instruction}
    if image is not None:
        payload["input_references"] = [
            {"type": "image_url", "image_url": {"url": pil_to_data_url(image)}}
        ]
    headers = {"Authorization": f"Bearer {_api_key()}",
               "Content-Type": "application/json"}

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            r = requests.post(IMAGES_API_URL, json=payload, headers=headers,
                              timeout=300)
            latency_s = round(time.time() - t0, 2)

            if not r.ok:
                try:
                    detail = r.json().get("error", {}).get("message", r.text[:300])
                except Exception:
                    detail = r.text[:300]
                raise RuntimeError(f"Images API error {r.status_code}: {detail}")

            data = r.json()
            images = data.get("data") or []
            usage = data.get("usage") or {}
            cost = float(usage.get("cost") or 0.0)

            meta = {
                "model": model, "instruction": instruction,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "cost_usd": cost, "latency_s": latency_s,
                "text_reply": "", "attempt": attempt,
            }
            _log_cost(model, run_name, step, cost,
                      usage.get("prompt_tokens", ""),
                      usage.get("completion_tokens", ""),
                      latency_s=latency_s,
                      note="" if images else "no image returned")

            if not images:
                raise RuntimeError("No image returned from Images API")

            b64 = images[0]["b64_json"]
            edited = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            return edited, meta

        except BudgetExceeded:
            raise
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(3 * attempt)

    raise RuntimeError(f"Edit failed after {max_retries} attempts: {last_err}")


def edit_image(image, instruction: str, model: str,
               run_name: str = "", step: str = "", max_retries: int = 3):
    """Apply one edit instruction to one image, OR generate a brand new image
    from text alone if image is None (used by generate_image below).

    Returns (edited PIL image, metadata dict). Raises BudgetExceeded when the
    spending cap is hit, or RuntimeError after repeated failures.
    """
    if model in IMAGES_API_MODELS:
        return _edit_image_via_images_api(image, instruction, model,
                                          run_name, step, max_retries)

    if spent_so_far() >= config.MAX_BUDGET_USD:
        raise BudgetExceeded(
            f"Spent ${spent_so_far():.2f} already, cap is "
            f"${config.MAX_BUDGET_USD:.2f}. Raise MAX_BUDGET_USD in "
            f"config.py only if you are sure."
        )

    content = [{"type": "text", "text": instruction}]
    if image is not None:
        content.append({"type": "image_url",
                        "image_url": {"url": pil_to_data_url(image)}})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        # Tell OpenRouter we want a picture back, not just words.
        "modalities": ["image", "text"],
        # Ask for the exact receipt: the response includes the true USD cost.
        "usage": {"include": True},
    }
    headers = {"Authorization": f"Bearer {_api_key()}",
               "Content-Type": "application/json"}

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            r = requests.post(API_URL, json=payload, headers=headers,
                              timeout=300)
            r.raise_for_status()
            data = r.json()
            latency_s = round(time.time() - t0, 2)
            if "error" in data:
                raise RuntimeError(f"API error: {data['error']}")

            msg = data["choices"][0]["message"]
            images = msg.get("images") or []
            usage = data.get("usage") or {}
            cost = float(usage.get("cost") or 0.0)

            meta = {
                "model": model,
                "instruction": instruction,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "cost_usd": cost,
                "latency_s": latency_s,
                "text_reply": (msg.get("content") or "")[:500],
                "attempt": attempt,
            }
            # We pay even when the model returns no image, so log first.
            _log_cost(model, run_name, step, cost,
                      usage.get("prompt_tokens", ""),
                      usage.get("completion_tokens", ""),
                      latency_s=latency_s,
                      note="" if images else "no image returned")

            if not images:
                # Usually a refusal or a text-only answer. Retrying often helps.
                raise RuntimeError(
                    f"No image returned. Model said: {meta['text_reply'][:200]}"
                )

            return data_url_to_pil(images[0]["image_url"]["url"]), meta

        except BudgetExceeded:
            raise
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(3 * attempt)

    raise RuntimeError(f"Edit failed after {max_retries} attempts: {last_err}")


def generate_image(prompt: str, model: str, run_name: str = "", step: str = "",
                   max_retries: int = 3):
    """Text-to-image: create a brand NEW image from a prompt alone, no input
    image. Thin wrapper around edit_image(image=None, ...) so it shares all
    the same cost logging, budget cap, and retry behavior."""
    return edit_image(None, prompt, model, run_name=run_name, step=step,
                      max_retries=max_retries)


if __name__ == "__main__":
    key_ok = "yes" if os.getenv("OPENROUTER_API_KEY") else "NO - put it in .env"
    print(f"key found: {key_ok}")
    print(f"spent so far: ${spent_so_far():.2f} of ${config.MAX_BUDGET_USD:.2f}")