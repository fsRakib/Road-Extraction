"""
Classical baseline. No weights, no GPU, no download - it runs today.

Uses a ridge filter (Frangi), originally built to find blood vessels in
medical scans. Roads are also long thin bright ridges, so it half-works.

Purpose: prove the whole pipeline end to end before you fight with PyTorch.
Every deep model you add should beat this. If one does not, something is wrong.
"""
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import frangi

from models._base import RoadModel


class BaselineRidge(RoadModel):
    name = "baseline"
    description = "Frangi ridge filter, no training required"

    def predict(self, patch):
        gray = rgb2gray(patch)
        # sigmas roughly match road half-widths in pixels at 0.55 m/px
        resp = frangi(gray, sigmas=range(2, 12, 2), black_ridges=False)
        # Normalise by a high percentile, not by the maximum. One bright
        # artefact would otherwise push every real road down towards zero.
        hi = np.percentile(resp, 99.0)
        if hi > 0:
            resp = np.clip(resp / hi, 0.0, 1.0)
        return resp.astype(np.float32)
