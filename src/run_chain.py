"""The robot. Runs one chain of edits on one image and saves every step.

Three modes:
  full    the whole image goes through the model each edit (what real apps do)
  crop    only a cropped patch around the allowed region goes through the
          model; the patch is pasted back with a soft edge (treatment 2a)
  rebase  every step sends the pristine ORIGINAL plus the full list of
          instructions so far, so nothing is ever a copy of a copy
          (treatment 3, edit-session design)

Optional flags:
  --diff       after a full-frame edit, restore original pixels wherever no
               real change happened (treatment 2b, diff compositing)
  --gate       retry once when measured damage is too high (verify-and-retry)
  --best-of N  always sample N candidates per step, keep the least damaging
               (unconditional best-of-N; overrides --gate)
  --size N     working resolution, longest side in pixels (resolution
               management experiments)
  --mock       use a free, offline stand-in instead of the real API. Every
               output is stamped "MOCK EDIT" in bright pink so it can never be
               mistaken for a real result. Use this only to rehearse the
               pipeline (does every mode/flag run without crashing?) before
               spending real credits. Never use mock output in the report.

Usage:
  python src/run_chain.py --image data/inputs/base.png \
      --script edit_scripts/chain_plain.json --mode full
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

import compositing
import config
import metrics
from or_client import edit_image as openrouter_edit_image
from gemini_client import edit_image as gemini_edit_image


def load_working_image(path, size):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    return img


def mock_edit_image(image, instruction, model="mock-local", run_name="", step=""):
    """A $0, offline stand-in for or_client.edit_image. Same inputs, same
    output shape (edited PIL image, meta dict), so run_chain.py cannot tell
    the difference. It softens the image a little (like a real re-render
    would) and stamps a bright, unmistakable MOCK EDIT label so these images
    can never be confused with real results. Use this only to rehearse the
    plumbing (does --mode crop/rebase, --diff, --gate, --best-of all run
    without crashing?) before spending real API credits.
    """
    arr = np.array(image).astype(np.float32)
    rng = np.random.default_rng()
    out = cv2.GaussianBlur(arr, (0, 0), 0.6)
    out = np.clip(out + rng.normal(0, 1.5, arr.shape), 0, 255).astype(np.uint8)
    h, w = out.shape[:2]
    bar_y0, bar_y1 = int(h * 0.75), int(h * 0.95)
    cv2.rectangle(out, (int(w * 0.05), bar_y0), (int(w * 0.95), bar_y1),
                  (255, 0, 200), -1)
    pil = Image.fromarray(out)
    d = ImageDraw.Draw(pil)
    d.text((int(w * 0.08), bar_y0 + 6), f"MOCK EDIT [{step}]", fill=(255, 255, 255))
    d.text((int(w * 0.08), bar_y0 + 26), instruction[:60], fill=(255, 255, 255))
    time.sleep(0.05)  # tiny fake delay so latency_s is never exactly 0
    meta = {
        "model": model, "instruction": instruction,
        "prompt_tokens": 0, "completion_tokens": 0,
        "cost_usd": 0.0, "latency_s": 0.05,
        "text_reply": "[mock] no real API called", "attempt": 1,
    }
    return pil, meta


def one_edit(current_np, original_np, step, all_instructions, mode, model,
             run_name, tag, editor):
    """Produce one edited frame according to the mode. Returns (image, meta).
    editor is the function that actually performs one edit call: either the
    real or_client.edit_image or the offline mock_edit_image above."""
    instruction = step["instruction"]
    regions = step.get("regions") or []
    scope = step.get("scope", "local")

    if mode == "rebase":
        text = ("Apply ALL of the following edits to this image:\n"
                + "\n".join(f"{i}. {t}" for i, t in enumerate(all_instructions, 1)))
        edited, meta = editor(Image.fromarray(original_np), text, model,
                              run_name=run_name, step=tag)
        meta["returned_size"] = list(edited.size)
        edited = edited.resize((original_np.shape[1], original_np.shape[0]),
                               Image.LANCZOS)
        meta["mode_note"] = f"rebase with {len(all_instructions)} instructions"
        return np.array(edited), meta

    if mode == "crop" and scope == "local" and regions:
        x0, y0, x1, y1 = compositing.union_bbox(regions, current_np.shape)
        patch = Image.fromarray(current_np[y0:y1, x0:x1])
        edited, meta = editor(patch, instruction, model,
                              run_name=run_name, step=tag)
        meta["returned_size"] = list(edited.size)
        edited = edited.resize((x1 - x0, y1 - y0), Image.LANCZOS)
        out = compositing.feather_paste(current_np, np.array(edited),
                                        (x0, y0, x1, y1))
        meta["mode_note"] = f"crop {x1 - x0}x{y1 - y0} at ({x0},{y0})"
        return out, meta

    # Full frame. Also the fallback in crop mode for global or region-less steps.
    edited, meta = editor(Image.fromarray(current_np), instruction, model,
                          run_name=run_name, step=tag)
    meta["returned_size"] = list(edited.size)
    edited = edited.resize((current_np.shape[1], current_np.shape[0]),
                           Image.LANCZOS)
    if mode == "crop":
        meta["mode_note"] = "crop mode fell back to full frame"
    return np.array(edited), meta


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--script", required=True)
    ap.add_argument("--mode", choices=["full", "crop", "rebase"], default="full")
    ap.add_argument("--model", default=config.PRIMARY_MODEL)
    ap.add_argument("--diff", action="store_true",
                    help="diff compositing after each full-frame edit")
    ap.add_argument("--gate", action="store_true",
                    help="retry an edit once when measured damage is too high")
    ap.add_argument("--gate-ssim", type=float, default=0.90)
    ap.add_argument("--best-of", type=int, default=1, dest="best_of",
                    help="sample N candidates per step, keep the least damaging")
    ap.add_argument("--size", type=int, default=config.WORKING_LONG_SIDE,
                    help="working resolution, longest side in pixels")
    ap.add_argument("--mock", action="store_true",
                    help="rehearse the pipeline for $0, no real API calls")
    ap.add_argument("--provider", choices=["openrouter", "gemini"],
                    default="openrouter",
                    help="which real API to call when not using --mock. "
                         "gemini = your free direct Gemini API key")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    if args.mock:
        editor = mock_edit_image
    elif args.provider == "gemini":
        editor = gemini_edit_image
    else:
        editor = openrouter_edit_image

    script = json.loads(Path(args.script).read_text())
    steps = script["steps"]

    stamp = datetime.now().strftime("%m%d_%H%M")
    suffix = args.mode + ("_diff" if args.diff else "") \
        + ("_gate" if args.gate else "") \
        + (f"_best{args.best_of}" if args.best_of > 1 else "") \
        + (f"_{args.size}px" if args.size != config.WORKING_LONG_SIDE else "") \
        + ("_MOCK" if args.mock else "")
    run_name = args.run_name or (
        f"{Path(args.image).stem}_{script.get('name', 'chain')}_{suffix}_{stamp}")
    if args.mock and args.run_name and not args.run_name.endswith("_MOCK"):
        run_name = args.run_name + "_MOCK"  # never silently overwrite a real run
    run_dir = config.RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "script.json").write_text(json.dumps(script, indent=2))

    img = load_working_image(args.image, args.size)
    img.save(run_dir / "step_00.png")
    original_np = np.array(img)
    current_np = original_np.copy()
    print(f"step_00 saved ({img.size[0]}x{img.size[1]}). "
          f"Mode={args.mode}, model={args.model}"
          + (" [MOCK - no real API calls, $0]" if args.mock else ""))

    manifest = {"image": str(args.image), "model": args.model,
                "mode": args.mode, "diff": args.diff, "gate": args.gate,
                "best_of": args.best_of, "size": args.size, "mock": args.mock,
                "script": script.get("name", ""),
                "started": datetime.now(timezone.utc).isoformat(), "steps": []}
    done_instructions = []

    for i, step in enumerate(steps, start=1):
        tag = f"step_{i:02d}"
        done_instructions.append(step["instruction"])
        print(f"[{tag}] {step['instruction'][:70]}")
        regions = step.get("regions") or []
        allowed = metrics.boxes_to_mask(regions, current_np.shape) if regions else None

        def produce(suffix=""):
            cand, m = one_edit(current_np, original_np, step,
                               done_instructions, args.mode, args.model,
                               run_name, tag + suffix, editor)
            if args.diff and args.mode == "full":
                cand, kept = compositing.diff_composite(current_np, cand, allowed)
                m["diff_kept_frac"] = round(kept, 4)
            return cand, m

        def damage_score(cand):
            gate_mask = allowed if allowed is not None else \
                metrics.change_mask(current_np, cand)
            aligned, _ = metrics.align(current_np, cand)
            return metrics.degradation_metrics(current_np, aligned,
                                               gate_mask)["ssim"]

        try:
            if args.best_of > 1:
                candidate, meta = produce()
                best_score = damage_score(candidate)
                scores = [round(best_score, 4)]
                extra_cost = 0.0
                for k in range(2, args.best_of + 1):
                    cand_k, meta_k = produce(f"_s{k}")
                    extra_cost += meta_k.get("cost_usd") or 0
                    s_k = damage_score(cand_k)
                    scores.append(round(s_k, 4))
                    if s_k > best_score:
                        best_score, candidate = s_k, cand_k
                meta["sample_ssims"] = scores
                meta["extra_cost_usd"] = round(extra_cost, 6)
            else:
                candidate, meta = produce()
                if args.gate:
                    score = damage_score(candidate)
                    meta["gate_ssim"] = round(score, 4)
                    if score < args.gate_ssim:
                        print(f"  gate: ssim {score:.3f} < {args.gate_ssim}, "
                              f"retrying once")
                        cand2, meta2 = produce("_retry")
                        score2 = damage_score(cand2)
                        meta["gate_retry_ssim"] = round(score2, 4)
                        meta["extra_cost_usd"] = meta2.get("cost_usd", 0)
                        if score2 > score:
                            candidate = cand2
        except Exception as e:
            print(f"  FAILED: {e}")
            manifest["steps"].append({"step": tag,
                                      "instruction": step["instruction"],
                                      "error": str(e)})
            break

        Image.fromarray(candidate).save(run_dir / f"{tag}.png")
        manifest["steps"].append({"step": tag, **meta})
        current_np = candidate

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum((s.get("cost_usd") or 0) + (s.get("extra_cost_usd") or 0)
                for s in manifest["steps"])
    print(f"\nDone. Outputs in {run_dir}")
    print(f"This run cost ${total:.3f}. "
          f"Next: python src/analyze_run.py --run {run_dir}")


if __name__ == "__main__":
    main()
