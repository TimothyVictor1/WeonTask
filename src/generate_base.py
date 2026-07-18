"""Generate the base photoshoot image from prompts/base_image_prompt.txt via
a real image-GENERATION API call (no input photo - this creates one from
scratch, unlike run_chain.py which edits an existing photo).

Saves N candidates so you can pick the best one - same idea as generating a
few options in the Gemini app, except scripted, cost-logged, and repeatable.

Usage:
    python src/generate_base.py --mock                 # free rehearsal first
    python src/generate_base.py --provider gemini --n 3
    python src/generate_base.py --provider openrouter --n 3

Then open the printed folder, pick your favorite, and:
    cp data/runs/base_candidates_.../candidate_02.png data/inputs/base.png
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

import config


def mock_generate_image(prompt, model="mock-local", run_name="", step=""):
    """Free, offline stand-in so you can test this script's flow (folders,
    numbering, printouts) before spending anything real."""
    img = Image.new("RGB", (768, 1024), (200, 200, 200))
    d = ImageDraw.Draw(img)
    d.text((20, 20), "MOCK BASE IMAGE", fill=(20, 20, 20))
    d.text((20, 50), prompt[:70], fill=(20, 20, 20))
    time.sleep(0.05)
    return img, {"model": model, "cost_usd": 0.0, "latency_s": 0.05}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt-file", default="prompts/base_image_prompt.txt")
    ap.add_argument("--provider", choices=["openrouter", "gemini"], default="gemini")
    ap.add_argument("--model", default=None,
                    help="default: OpenRouter uses google/gemini-3.1-flash-image, "
                         "Gemini direct uses gemini-3.1-flash-image")
    ap.add_argument("--n", type=int, default=3, help="how many candidates to generate")
    ap.add_argument("--mock", action="store_true",
                    help="rehearse this script for $0, no real API calls")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    prompt = Path(args.prompt_file).read_text().strip()

    if args.mock:
        gen, model = mock_generate_image, (args.model or "mock")
    elif args.provider == "gemini":
        from gemini_client import generate_image as gen
        model = args.model or "gemini-3.1-flash-image"
    else:
        from or_client import generate_image as gen
        model = args.model or config.PRIMARY_MODEL

    stamp = datetime.now().strftime("%m%d_%H%M")
    suffix = f"{args.provider}_{stamp}" + ("_MOCK" if args.mock else "")
    out_dir = Path(args.out_dir) if args.out_dir else \
        config.RUNS_DIR / f"base_candidates_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n} candidate(s) with {model}"
          + (" [MOCK - $0]" if args.mock else "") + " ...")
    total_cost = 0.0
    for i in range(1, args.n + 1):
        img, meta = gen(prompt, model, run_name=out_dir.name, step=f"cand_{i:02d}")
        path = out_dir / f"candidate_{i:02d}.png"
        img.save(path)
        cost = meta.get("cost_usd") or 0.0
        total_cost += cost
        print(f"  [{i}/{args.n}] saved {path.name} "
              f"({img.size[0]}x{img.size[1]}, ${cost:.4f})")

    print(f"\nDone. {args.n} candidates in {out_dir}")
    print(f"Total cost: ${total_cost:.4f}")
    print("Open them, pick your favorite, then run:")
    print(f"  cp {out_dir}/candidate_01.png data/inputs/base.png")
    print("(swap 01 for whichever candidate number you liked best)")


if __name__ == "__main__":
    main()
