"""Read a run folder produced by run_chain.py and answer one question:

    how much did the parts we never asked to change get damaged, step by step?

Intended edit regions come from the script.json that run_chain.py copies into
every run folder. On top of the standard masked metrics this also reports:

  near vs far   damage in a 40 px ring around the edited regions vs the far
                field, answering "where does damage land, near the edit or
                everywhere?"
  watch boxes   optional named regions in the script ("watch_regions"), e.g.
                the logo, the sign, the face, each measured per step against
                its reference step. The face entry uses step 1 as reference:
                once the user approves the smile, the face should never
                change again. If the optional insightface library is
                installed, a true face-embedding similarity column is added.

Writes metrics.csv, curve.png, strip.png, detail_strip.png into the run folder.

Usage:
    python src/analyze_run.py --run data/runs/<run_folder>
"""
import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

import metrics as M

try:
    import face_embed
    HAVE_FACE_EMBED = face_embed.available()
except Exception:
    HAVE_FACE_EMBED = False

NEAR_PX = 40  # width of the "near the edit" ring, in pixels


def load_steps(run_dir: Path):
    files = sorted(run_dir.glob("step_*.png"))
    if len(files) < 2:
        raise SystemExit(f"Need at least step_00 and step_01 in {run_dir}")
    imgs = [np.array(Image.open(f).convert("RGB")) for f in files]
    names = [f.stem for f in files]
    return names, imgs


def load_costs(run_dir: Path):
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        return {}
    data = json.loads(manifest.read_text())
    return {s.get("step"): s.get("cost_usd") for s in data.get("steps", [])}


def load_script(run_dir: Path, override):
    p = Path(override) if override else run_dir / "script.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def parse_rel_box(text: str):
    x, y, w, h = (float(v) for v in text.split(","))
    return [x, y, w, h]


def crop_box(img, box):
    h, w = img.shape[:2]
    x, y, bw, bh = box
    return img[int(y * h):int((y + bh) * h), int(x * w):int((x + bw) * w)]


def make_strip(imgs, names, out_path: Path, thumb_h: int = 300, crop=None):
    tiles = []
    for img, name in zip(imgs, names):
        h, w = img.shape[:2]
        if crop is not None:
            x, y, cw, ch = crop
            x0, y0 = int(x * w), int(y * h)
            x1, y1 = min(int((x + cw) * w), w), min(int((y + ch) * h), h)
            img = img[y0:y1, x0:x1]
            h, w = img.shape[:2]
        scale = thumb_h / h
        tile = Image.fromarray(img).resize(
            (max(int(w * scale), 1), thumb_h), Image.LANCZOS)
        canvas = Image.new("RGB", (tile.width, thumb_h + 28), "white")
        canvas.paste(tile, (0, 0))
        ImageDraw.Draw(canvas).text((6, thumb_h + 6), name, fill="black")
        tiles.append(canvas)
    total_w = sum(t.width for t in tiles) + 4 * (len(tiles) - 1)
    strip = Image.new("RGB", (total_w, thumb_h + 28), "white")
    x = 0
    for t in tiles:
        strip.paste(t, (x, 0))
        x += t.width + 4
    strip.save(out_path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder under data/runs")
    ap.add_argument("--script", default=None,
                    help="script JSON with regions; default <run>/script.json")
    ap.add_argument("--detail", default="0.30,0.05,0.40,0.40",
                    help="relative x,y,w,h crop for the zoomed detail strip")
    args = ap.parse_args()

    run_dir = Path(args.run)
    names, imgs = load_steps(run_dir)
    costs = load_costs(run_dir)
    script = load_script(run_dir, args.script)
    script_steps = (script or {}).get("steps", [])
    watch = (script or {}).get("watch_regions", [])

    ref0 = imgs[0]
    shape = ref0.shape

    aligned = [ref0]
    shifts = [0.0]
    for img in imgs[1:]:
        a, s = M.align(ref0, img)
        aligned.append(a)
        shifts.append(s)

    # Cumulative damage at step i is measured vs step 0 OUTSIDE the union of
    # every intended region so far: those pixels were never part of ANY
    # requested edit, so ideally they are still identical to the original.
    union = np.zeros(shape[:2], dtype=bool)
    kernel = np.ones((2 * NEAR_PX + 1, 2 * NEAR_PX + 1), np.uint8)
    rows = []
    for i in range(1, len(aligned)):
        name = names[i]
        sdef = script_steps[i - 1] if i - 1 < len(script_steps) else {}
        regions = sdef.get("regions") or []
        if regions and sdef.get("scope", "local") == "local":
            intended = M.boxes_to_mask(regions, shape)
            mask_src = "boxes"
        else:
            intended = M.change_mask(aligned[i - 1], aligned[i])
            mask_src = "auto"
        union |= intended

        cum = M.degradation_metrics(ref0, aligned[i], union)
        inc = M.degradation_metrics(aligned[i - 1], aligned[i], intended)
        lp = M.lpips_outside(ref0, aligned[i], union)

        # Near vs far: a ring around everything ever edited, vs the far field.
        dil = cv2.dilate(union.astype(np.uint8), kernel).astype(bool)
        ring = dil & ~union
        far = ~dil
        near_m = M.degradation_metrics(ref0, aligned[i], ~ring)
        far_m = M.degradation_metrics(ref0, aligned[i], ~far)

        row = {
            "step": name,
            "shift_px": round(shifts[i], 2),
            "mask_src": mask_src,
            "edited_area_pct": round(float(intended.mean()) * 100, 2),
            "cum_ssim": round(cum["ssim"], 4),
            "cum_psnr": round(cum["psnr"], 2),
            "cum_delta_e": round(cum["delta_e"], 3),
            "cum_sharpness_ratio": round(cum["sharpness_ratio"], 4),
            "cum_collateral_pct": round(cum["collateral_pct"], 2),
            "cum_lpips": round(lp, 4) if not math.isnan(lp) else "",
            "near_ssim": round(near_m["ssim"], 4),
            "near_delta_e": round(near_m["delta_e"], 3),
            "far_ssim": round(far_m["ssim"], 4),
            "far_delta_e": round(far_m["delta_e"], 3),
            "inc_ssim": round(inc["ssim"], 4),
            "inc_delta_e": round(inc["delta_e"], 3),
            "cost_usd": costs.get(name, ""),
        }

        # Watch boxes: named regions measured against their reference step.
        empty = np.zeros((2, 2), bool)
        for wr in watch:
            ref_idx = int(wr.get("from", 0))
            key_s = f"{wr['name']}_ssim"
            key_d = f"{wr['name']}_delta_e"
            key_e = f"{wr['name']}_embed_sim"
            if i > ref_idx:
                a = crop_box(aligned[ref_idx], wr["box"])
                b = crop_box(aligned[i], wr["box"])
                no_mask = np.zeros(a.shape[:2], bool)
                wm = M.degradation_metrics(a, b, no_mask)
                row[key_s] = round(wm["ssim"], 4)
                row[key_d] = round(wm["delta_e"], 3)
                if HAVE_FACE_EMBED and wr["name"] == "face":
                    sim = face_embed.similarity(a, b)
                    row[key_e] = round(sim, 4) if sim is not None else ""
                elif wr["name"] == "face":
                    row[key_e] = ""
            else:
                row[key_s] = ""
                row[key_d] = ""
                if wr["name"] == "face":
                    row[key_e] = ""
        rows.append(row)
        print(f"{name}: cum SSIM {row['cum_ssim']}, "
              f"cum deltaE {row['cum_delta_e']}, "
              f"near dE {row['near_delta_e']} vs far dE {row['far_delta_e']}, "
              f"shift {row['shift_px']}px ({mask_src} mask)")

    out_csv = run_dir / "metrics.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    xs = list(range(1, len(rows) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(xs, [r["cum_ssim"] for r in rows], marker="o", color="#c0392b")
    axes[0].set_title("Structure preserved outside edits (SSIM)")
    axes[0].set_xlabel("edits applied")
    axes[0].set_ylabel("SSIM vs original (1.0 = untouched)")
    axes[0].grid(alpha=0.3)
    axes[1].plot(xs, [r["cum_delta_e"] for r in rows], marker="o",
                 color="#2c3e50", label="mean color drift (delta E)")
    axes[1].plot(xs, [r["cum_collateral_pct"] for r in rows], marker="s",
                 color="#7f8c8d", label="% pixels visibly changed")
    axes[1].plot(xs, [r["near_delta_e"] for r in rows], marker="^",
                 color="#e67e22", label="delta E near edits")
    axes[1].plot(xs, [r["far_delta_e"] for r in rows], marker="v",
                 color="#16a085", label="delta E far from edits")
    axes[1].set_title("Color drift, collateral, near vs far")
    axes[1].set_xlabel("edits applied")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"Degradation over the edit chain: {run_dir.name}")
    fig.tight_layout()
    fig.savefig(run_dir / "curve.png", dpi=150)

    make_strip(imgs, names, run_dir / "strip.png")
    make_strip(imgs, names, run_dir / "detail_strip.png",
               crop=parse_rel_box(args.detail))

    print(f"\nWrote {out_csv}, curve.png, strip.png, detail_strip.png "
          f"in {run_dir}")
    if not HAVE_FACE_EMBED:
        print("note: face embedding column empty "
              "(optional: pip3 install insightface onnxruntime)")


if __name__ == "__main__":
    main()
