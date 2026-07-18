"""Optional true face-identity measurement using ArcFace embeddings.

Everything works without this: analyze_run.py already measures face drift with
SSIM and delta E on the face watch box. Install the extras to also get a real
face-recognition similarity column (1.0 = same identity, lower = drifting):

    pip3 install insightface onnxruntime

The first run downloads the model (a few hundred MB, one time). If anything
here fails, the pipeline continues without it.
"""
import numpy as np

_app = None
_ok = None


def available() -> bool:
    """True when insightface is installed and its model loads."""
    global _app, _ok
    if _ok is not None:
        return _ok
    try:
        from insightface.app import FaceAnalysis
        _app = FaceAnalysis(name="buffalo_l",
                            providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=-1, det_size=(640, 640))
        _ok = True
    except Exception:
        _app = None
        _ok = False
    return _ok


def _embed(img_rgb: np.ndarray):
    """Embedding of the largest face in the crop, or None."""
    if not available():
        return None
    try:
        faces = _app.get(img_rgb[:, :, ::-1])  # insightface expects BGR
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0])
                   * (f.bbox[3] - f.bbox[1]))
        v = face.normed_embedding
        return v / (np.linalg.norm(v) + 1e-9)
    except Exception:
        return None


def similarity(img_a: np.ndarray, img_b: np.ndarray):
    """Cosine similarity between the faces in two crops, or None."""
    ea, eb = _embed(img_a), _embed(img_b)
    if ea is None or eb is None:
        return None
    return float(np.dot(ea, eb))
