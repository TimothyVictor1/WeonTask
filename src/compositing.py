"""Treatment 2 tools: pipeline design around the edit.

Two ideas from Weon's brief, implemented:

1. crop-edit-composite: cut out just the region the edit needs (plus some
   context), let the model edit only that patch, then paste the patch back
   into the untouched previous image with a soft feathered edge. Pixels
   outside the patch never pass through the model at all, so they cannot
   degrade.

2. diff compositing: after a full-frame edit, detect what actually changed,
   keep the edited pixels only there (plus the allowed region), and restore
   the original pixels everywhere else. Kills slow global drift.
"""
import cv2
import numpy as np

import config
import metrics


def union_bbox(regions, shape, pad_frac=None):
    """Smallest pixel rectangle covering all allowed boxes, plus padding so the
    model gets context. regions are relative [x, y, w, h]. Returns x0,y0,x1,y1."""
    if pad_frac is None:
        pad_frac = config.CROP_PAD_FRAC
    h, w = shape[:2]
    xs0, ys0, xs1, ys1 = [], [], [], []
    for (x, y, bw, bh) in regions:
        xs0.append(x * w)
        ys0.append(y * h)
        xs1.append((x + bw) * w)
        ys1.append((y + bh) * h)
    x0, y0, x1, y1 = min(xs0), min(ys0), max(xs1), max(ys1)
    pad = pad_frac * max(x1 - x0, y1 - y0)
    x0 = int(max(0, x0 - pad))
    y0 = int(max(0, y0 - pad))
    x1 = int(min(w, x1 + pad))
    y1 = int(min(h, y1 + pad))
    # Give the model at least some surroundings to reason about.
    min_side = 256
    if x1 - x0 < min_side:
        cx = (x0 + x1) // 2
        x0 = max(0, cx - min_side // 2)
        x1 = min(w, x0 + min_side)
    if y1 - y0 < min_side:
        cy = (y0 + y1) // 2
        y0 = max(0, cy - min_side // 2)
        y1 = min(h, y0 + min_side)
    return x0, y0, x1, y1


def feather_paste(base, patch, bbox, feather_px=None):
    """Paste patch into base at bbox with a soft edge so there is no visible
    seam. base and patch are numpy RGB arrays; patch must match bbox size."""
    if feather_px is None:
        feather_px = config.FEATHER_PX
    x0, y0, x1, y1 = bbox
    out = base.astype(np.float32).copy()
    placed = out.copy()
    placed[y0:y1, x0:x1] = patch.astype(np.float32)

    f = int(min(feather_px, (x1 - x0) // 4, (y1 - y0) // 4))
    f = max(f, 1)
    alpha = np.zeros(base.shape[:2], np.float32)
    alpha[y0 + f:y1 - f, x0 + f:x1 - f] = 1.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=f / 2 + 0.1)
    alpha = alpha[..., None]
    return (out * (1 - alpha) + placed * alpha).clip(0, 255).astype(np.uint8)


def diff_composite(prev, edited, allowed_mask=None, thresh=10.0):
    """Keep edited pixels only where a real change happened (or was allowed);
    restore prev pixels everywhere else, with a feathered blend.

    Returns (result, kept_fraction): kept_fraction is how much of the frame
    kept the model's new pixels.
    """
    aligned, _shift = metrics.align(prev, edited)
    keep = metrics.change_mask(prev, aligned, thresh=thresh, dilate_px=12)
    if allowed_mask is not None:
        keep = keep | allowed_mask
    alpha = cv2.GaussianBlur(keep.astype(np.float32), (0, 0),
                             sigmaX=max(config.FEATHER_PX / 2, 1))[..., None]
    out = (prev.astype(np.float32) * (1 - alpha)
           + aligned.astype(np.float32) * alpha)
    return out.clip(0, 255).astype(np.uint8), float(keep.mean())
