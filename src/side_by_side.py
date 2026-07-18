"""Stack runs on top of each other: one row per run, one column per step.
This is the "same 5-edit chain with and without your approach, side by side
at each step" figure from the brief.

Usage:
    python src/side_by_side.py --runs data/runs/plain data/runs/crop \
        --labels plain crop --out data/runs/side_by_side.png

Optional --crop x,y,w,h (relative) zooms every tile to the same region.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

LABEL_W = 110


def load_steps(run_dir: Path):
    files = sorted(Path(run_dir).glob("step_*.png"))
    if len(files) < 2:
        raise SystemExit(f"Need steps in {run_dir}")
    return [np.array(Image.open(f).convert("RGB")) for f in files]


def tile(img, thumb_h, crop):
    h, w = img.shape[:2]
    if crop is not None:
        x, y, cw, ch = crop
        x0, y0 = int(x * w), int(y * h)
        x1, y1 = min(int((x + cw) * w), w), min(int((y + ch) * h), h)
        img = img[y0:y1, x0:x1]
        h, w = img.shape[:2]
    scale = thumb_h / h
    return Image.fromarray(img).resize(
        (max(int(w * scale), 1), thumb_h), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--out", default="data/runs/side_by_side.png")
    ap.add_argument("--thumb", type=int, default=240)
    ap.add_argument("--crop", default=None,
                    help="relative x,y,w,h to zoom every tile")
    args = ap.parse_args()
    labels = args.labels or [Path(r).name for r in args.runs]
    crop = None
    if args.crop:
        crop = [float(v) for v in args.crop.split(",")]

    all_steps = [load_steps(r) for r in args.runs]
    n_cols = min(len(s) for s in all_steps)

    rows_img = []
    for steps, label in zip(all_steps, labels):
        tiles = [tile(steps[c], args.thumb, crop) for c in range(n_cols)]
        row_w = LABEL_W + sum(t.width for t in tiles) + 4 * (n_cols - 1)
        row = Image.new("RGB", (row_w, args.thumb + 24), "white")
        d = ImageDraw.Draw(row)
        d.text((8, args.thumb // 2), label, fill="black")
        x = LABEL_W
        for c, t in enumerate(tiles):
            row.paste(t, (x, 0))
            d.text((x + 4, args.thumb + 6), f"step_{c:02d}", fill="black")
            x += t.width + 4
        rows_img.append(row)

    total_w = max(r.width for r in rows_img)
    total_h = sum(r.height for r in rows_img) + 8 * (len(rows_img) - 1)
    grid = Image.new("RGB", (total_w, total_h), "white")
    y = 0
    for r in rows_img:
        grid.paste(r, (0, y))
        y += r.height + 8
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
