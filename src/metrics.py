"""The measuring tools. Everything here answers one question:

    How different are two images in the places that were NEVER meant to change?

Key idea: for every edit we build a mask (a stencil) marking the region the edit
was supposed to touch. Everything outside that stencil should come back
pixel-identical. Any difference out there is damage, and we can measure it with
standard image-comparison math because we have the previous image as ground truth.
"""
import cv2
import numpy as np
from skimage.color import deltaE_ciede2000
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation

# LPIPS is an optional extra (needs torch, a heavy install). Everything works without it.
try:
    import lpips as _lpips  # type: ignore
    import torch  # type: ignore
    _LPIPS_NET = None
    HAVE_LPIPS = True
except ImportError:
    HAVE_LPIPS = False


def rgb2lab(rgb_uint8: np.ndarray) -> np.ndarray:
    """RGB (uint8, 0-255) -> CIE Lab, numerically matching skimage's own
    rgb2lab (max difference ~0.0002, far below any threshold we use).

    Hand-written with only elementwise multiply/add - never a matrix
    multiply. skimage's rgb2lab internally does `arr @ matrix.T` to convert
    RGB to XYZ, and on some Macs that exact operation crashes the whole
    process (a hard, uncatchable segfault, not a Python exception) due to a
    bug in how numpy's default BLAS backend (Apple's Accelerate framework)
    handles that matmul. Since color difference is computed for nearly every
    metric in this project, routing around it here removes the single
    biggest crash risk in the whole pipeline.
    """
    arr = rgb_uint8.astype(np.float64) / 255.0
    mask = arr > 0.04045
    arr_lin = np.where(mask, ((arr + 0.055) / 1.055) ** 2.4, arr / 12.92)
    r, g, b = arr_lin[..., 0], arr_lin[..., 1], arr_lin[..., 2]
    x = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z = 0.019334 * r + 0.119193 * g + 0.950227 * b
    xn, yn, zn = 0.95047, 1.0, 1.08883
    x, y, z = x / xn, y / yn, z / zn
    eps = 216 / 24389
    kappa = 24389 / 27
    fx = np.where(x > eps, np.cbrt(x), (kappa * x + 16) / 116)
    fy = np.where(y > eps, np.cbrt(y), (kappa * y + 16) / 116)
    fz = np.where(z > eps, np.cbrt(z), (kappa * z + 16) / 116)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    bb = 200 * (fy - fz)
    return np.stack([L, a, bb], axis=-1)


# ---------------------------------------------------------------- alignment

def align(ref: np.ndarray, img: np.ndarray, max_iters: int = 100):
    """Nudge img so it sits exactly on top of ref.

    Editing models sometimes shift the whole frame slightly. If we compared
    without aligning first, that shift would look like massive damage
    everywhere. Returns (aligned_img, shift_in_pixels). shift = -1.0 means
    alignment failed and we compared as-is; a large shift is itself a finding
    (the model recomposed the frame).

    Uses skimage's FFT-based phase correlation (translation only, whole-pixel
    precision) rather than OpenCV's cv2.findTransformECC. ECC's iterative
    solver is known to crash the whole process (a hard, uncatchable segfault,
    not a Python exception) on some machines - most often when a conda
    environment's numpy is linked against a different math library than the
    opencv-python wheel. Phase correlation with upsample_factor=1 uses only
    FFT and elementwise ops (no matrix multiply at all - skimage's own
    sub-pixel refinement does use one, which is why it's deliberately turned
    off here), so it never touches the same crash-prone code path. Whole-pixel
    precision is plenty: we only need to know whether a shift is ~0px or
    several pixels, not fractions of a pixel.
    """
    if img.shape != ref.shape:
        img = cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    g_ref = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    try:
        shift_yx, _error, _diffphase = phase_cross_correlation(
            g_ref, g_img, upsample_factor=1)
        dy, dx = float(shift_yx[0]), float(shift_yx[1])
        warp = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned = cv2.warpAffine(
            img, warp, (ref.shape[1], ref.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        shift = float(np.hypot(dx, dy))
        return aligned, shift
    except Exception:
        return img, -1.0


# ------------------------------------------------------------- edit masks

def change_mask(a: np.ndarray, b: np.ndarray, thresh: float = 12.0,
                min_area_frac: float = 0.0005, dilate_px: int = 15) -> np.ndarray:
    """Stencil of where b visibly differs from a.

    Used to approximate the intended edit region when no boxes are given.
    thresh is a CIEDE2000 color distance: about 2 is barely visible to the
    eye, 10 and up is an obvious change, so the default keeps only strong
    changes (the edit itself) and ignores the faint everywhere-noise (the
    degradation). Small specks are dropped and the result is padded outward
    (dilated) so the edit's soft boundary is safely inside the stencil.
    """
    de = deltaE_ciede2000(rgb2lab(a), rgb2lab(b)).astype(np.float32)
    m = (cv2.GaussianBlur(de, (5, 5), 0) > thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    min_area = min_area_frac * m.size
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 1

    if dilate_px > 0:
        keep = cv2.dilate(keep, np.ones((dilate_px, dilate_px), np.uint8))
    return keep.astype(bool)


def boxes_to_mask(boxes, shape, pad_px: int = 10) -> np.ndarray:
    """Turn hand-stated boxes into a stencil.

    boxes are relative coordinates [x, y, w, h] between 0 and 1, so they
    survive any resize. This is the rigorous option: you state where the
    edit was SUPPOSED to happen instead of inferring it from what changed.
    """
    h, w = shape[:2]
    m = np.zeros((h, w), dtype=bool)
    for (x, y, bw, bh) in boxes:
        x0 = max(int(x * w) - pad_px, 0)
        y0 = max(int(y * h) - pad_px, 0)
        x1 = min(int((x + bw) * w) + pad_px, w)
        y1 = min(int((y + bh) * h) + pad_px, h)
        m[y0:y1, x0:x1] = True
    return m


# ---------------------------------------------------------------- metrics

def degradation_metrics(ref: np.ndarray, img: np.ndarray, exclude_mask: np.ndarray) -> dict:
    """Compare img against ref only where exclude_mask is False.

    That area was never meant to change, so any difference there is damage.
    Returns:
      ssim            structure similarity, 1.0 = identical, drops as detail is lost
      psnr            pixel accuracy in dB, higher is better, 40+ is near identical
      delta_e         average perceptual color difference, below 2 is invisible
      sharpness_ratio fine-detail energy vs the reference, 1.0 = as sharp, <1 = softened
      collateral_pct  percent of untouched pixels that visibly changed (delta E > 4)
      outside_frac    how much of the image the untouched area covers (sanity info)
    """
    outside = ~exclude_mask
    if outside.sum() < 100:
        return {"ssim": float("nan"), "psnr": float("nan"), "delta_e": float("nan"),
                "sharpness_ratio": float("nan"), "collateral_pct": float("nan"),
                "outside_frac": 0.0}

    g_ref = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    g_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    _, smap = structural_similarity(g_ref, g_img, full=True, data_range=255)
    ssim_out = float(smap[outside].mean())

    diff = ref.astype(np.float32) - img.astype(np.float32)
    mse = float((diff[outside] ** 2).mean())
    psnr = float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))

    de = deltaE_ciede2000(rgb2lab(ref), rgb2lab(img))
    delta_e = float(de[outside].mean())
    collateral_pct = float((de[outside] > 4.0).mean() * 100)

    lap_ref = cv2.Laplacian(g_ref, cv2.CV_64F)
    lap_img = cv2.Laplacian(g_img, cv2.CV_64F)
    v_ref = float(lap_ref[outside].var())
    v_img = float(lap_img[outside].var())
    sharpness_ratio = float(v_img / v_ref) if v_ref > 0 else float("nan")

    return {"ssim": ssim_out, "psnr": psnr, "delta_e": delta_e,
            "sharpness_ratio": sharpness_ratio, "collateral_pct": collateral_pct,
            "outside_frac": float(outside.mean())}


def lpips_outside(ref: np.ndarray, img: np.ndarray, exclude_mask: np.ndarray) -> float:
    """Optional: a neural network's opinion of how different two images LOOK.

    We copy the reference pixels into the edited region first, so the score
    only reflects damage outside it. Lower is better, 0 means identical.
    Returns nan when lpips/torch are not installed.
    """
    if not HAVE_LPIPS:
        return float("nan")
    global _LPIPS_NET
    if _LPIPS_NET is None:
        _LPIPS_NET = _lpips.LPIPS(net="alex")

    test = img.copy()
    test[exclude_mask] = ref[exclude_mask]

    def to_tensor(x):
        return torch.from_numpy(x.copy()).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0

    with torch.no_grad():
        return float(_LPIPS_NET(to_tensor(ref), to_tensor(test)).item())