"""End-to-end sanity check. Costs nothing, needs no API key.

It fabricates a fake edit chain where the ground truth is known (each step
recolors one region and slightly blurs and noises the whole frame), then
verifies every part of the pipeline reads that damage correctly:

  1. the automatic change mask finds the edited region
  2. feather_paste changes nothing outside its box
  3. diff_composite restores untouched pixels after a noisy fake edit
  4. analyze_run.py produces a falling SSIM curve and a rising delta E
  5. compare_runs.py runs and saves a chart

Run from the project root:  python tests/test_synthetic.py
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import compositing as C  # noqa: E402
import metrics as M      # noqa: E402

RUN = ROOT / "data" / "runs" / "synthetic_demo"


def build_base(rng):
    """A textured test image: noise + shapes + text, so SSIM has detail."""
    base = np.clip(rng.normal(128, 22, (768, 1024, 3)), 0, 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 1.2)
    grad = np.linspace(0, 50, 1024, dtype=np.float32)[None, :, None]
    base = np.clip(base.astype(np.float32) + grad, 0, 255).astype(np.uint8)
    cv2.rectangle(base, (100, 100), (300, 340), (200, 60, 60), -1)
    cv2.circle(base, (520, 250), 90, (240, 220, 80), -1)
    cv2.rectangle(base, (700, 400), (930, 700), (60, 120, 200), -1)
    cv2.putText(base, "WEON DEMO 123", (120, 640),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
    return base


def degrade(img, rng):
    """What a lossy re-render does to untouched areas: soften + noise."""
    out = cv2.GaussianBlur(img, (0, 0), 0.7)
    out = np.clip(out.astype(np.float32) + rng.normal(0, 2.0, img.shape), 0, 255)
    return out.astype(np.uint8)


def main():
    rng = np.random.default_rng(7)
    RUN.mkdir(parents=True, exist_ok=True)

    base = build_base(rng)
    cv2.imwrite(str(RUN / "step_00.png"), cv2.cvtColor(base, cv2.COLOR_RGB2BGR))

    edits = [
        ((100, 100, 200, 240), (40, 170, 90)),
        ((430, 160, 180, 180), (200, 90, 200)),
        ((700, 400, 230, 300), (230, 140, 40)),
    ]
    steps = []
    cur = base
    for i, ((x, y, w, h), color) in enumerate(edits, start=1):
        step = degrade(cur, rng)
        step[y:y + h, x:x + w] = color
        cv2.imwrite(str(RUN / f"step_{i:02d}.png"),
                    cv2.cvtColor(step, cv2.COLOR_RGB2BGR))
        steps.append({"instruction": f"recolor region {i}", "scope": "local",
                      "regions": [[x / 1024, y / 768, w / 1024, h / 768]]})
        cur = step
    (RUN / "script.json").write_text(
        json.dumps({"name": "synthetic", "steps": steps}, indent=2))

    # 1) the automatic mask finds the edit on its own
    s1 = cv2.cvtColor(cv2.imread(str(RUN / "step_01.png")), cv2.COLOR_BGR2RGB)
    auto = M.change_mask(base, s1)
    truth = M.boxes_to_mask(steps[0]["regions"], base.shape, pad_px=0)
    iou = (auto & truth).sum() / max((auto | truth).sum(), 1)
    print(f"[1] auto-mask IoU vs ground truth: {iou:.2f}")
    assert iou > 0.5, "auto mask failed to find the edited region"

    # 2) feather_paste changes nothing outside its box
    x0, y0, x1, y1 = 300, 300, 600, 560
    patch = np.full((y1 - y0, x1 - x0, 3), 200, np.uint8)
    pasted = C.feather_paste(base, patch, (x0, y0, x1, y1))
    outside = np.ones(base.shape[:2], bool)
    outside[max(y0 - 32, 0):y1 + 32, max(x0 - 32, 0):x1 + 32] = False
    worst = int(np.abs(pasted.astype(int) - base.astype(int))[outside].max())
    print(f"[2] feather_paste max change outside box: {worst}")
    assert worst <= 1, "feather_paste leaked outside its box"

    # 3) diff_composite restores untouched pixels after a noisy fake edit
    fake = degrade(base, rng)
    fake[100:340, 100:300] = (10, 200, 30)
    fixed, kept = C.diff_composite(base, fake)
    outside = np.ones(base.shape[:2], bool)
    outside[40:400, 40:360] = False
    resid = int(np.abs(fixed.astype(int) - base.astype(int))[outside].max())
    print(f"[3] diff_composite residual outside edit: {resid} "
          f"(kept {kept:.1%} of frame from the edit)")
    assert resid <= 2, "diff_composite failed to restore untouched pixels"

    # 4) the real analysis CLI end to end
    r = subprocess.run(
        [sys.executable, "src/analyze_run.py", "--run", str(RUN),
         "--detail", "0.08,0.70,0.55,0.20"],
        cwd=ROOT, capture_output=True, text=True)
    print(r.stdout)
    assert r.returncode == 0, r.stderr

    with open(RUN / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    ssim = [float(r["cum_ssim"]) for r in rows]
    de = [float(r["cum_delta_e"]) for r in rows]
    assert all(a > b for a, b in zip(ssim, ssim[1:])), f"SSIM should fall: {ssim}"
    assert all(a < b for a, b in zip(de, de[1:])), f"delta E should rise: {de}"
    print("[4] degradation curve moves the right way")

    # 5) compare_runs smoke test
    r = subprocess.run(
        [sys.executable, "src/compare_runs.py", "--runs", str(RUN),
         "--labels", "synthetic", "--out", str(RUN / "cmp.png")],
        cwd=ROOT, capture_output=True, text=True)
    print(r.stdout)
    assert r.returncode == 0, r.stderr

    print("ALL CHECKS PASSED")
    print(f"SSIM per step:   {ssim}")
    print(f"deltaE per step: {de}")


if __name__ == "__main__":
    main()
