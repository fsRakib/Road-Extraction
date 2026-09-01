"""
Run a patch-based model over a big image without seams at patch edges.

A model only sees one 512x512 patch at a time, so a road running along a
patch boundary gets cut with no context on one side - the prediction is
weakest exactly at every seam. The fix costs no training: overlap the
patches and blend the overlapping region, weighted so each pixel is
influenced most by the patch where it sits closest to the centre (a Hann
window). Free accuracy, pretrained model unchanged.
"""
import numpy as np


def _hann_window(size):
    w1d = np.hanning(size + 2)[1:-1]           # avoid the zero endpoints
    return np.outer(w1d, w1d).astype(np.float32)


def tiled_predict(model, image, patch_px, overlap_px):
    """Run model.predict() over overlapping patches and blend the results."""
    h, w = image.shape[:2]
    stride = patch_px - overlap_px
    window = _hann_window(patch_px)

    acc = np.zeros((h, w), np.float32)
    weight = np.zeros((h, w), np.float32)

    rows = list(range(0, max(h - patch_px, 0) + 1, stride)) or [0]
    cols = list(range(0, max(w - patch_px, 0) + 1, stride)) or [0]
    if rows[-1] + patch_px < h:
        rows.append(h - patch_px)
    if cols[-1] + patch_px < w:
        cols.append(w - patch_px)

    for r in rows:
        for c in cols:
            tile = image[r:r + patch_px, c:c + patch_px]
            th, tw = tile.shape[:2]
            if (th, tw) != (patch_px, patch_px):
                pad = np.zeros((patch_px, patch_px, 3), tile.dtype)
                pad[:th, :tw] = tile
                tile = pad

            out = model.predict(tile)
            acc[r:r + th, c:c + tw] += (out * window)[:th, :tw]
            weight[r:r + th, c:c + tw] += window[:th, :tw]

    weight[weight == 0] = 1.0
    return (acc / weight).astype(np.float32)
