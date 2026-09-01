"""Overlay pictures so you can judge the result by eye."""
import numpy as np
from PIL import Image


def save_overlay(image_rgb, mask, path, color=(255, 40, 40), alpha=0.55):
    """Paint the predicted mask on top of the satellite image and save a PNG."""
    out = image_rgb.astype(np.float32).copy()
    tint = np.array(color, dtype=np.float32)
    sel = mask.astype(bool)
    out[sel] = (1.0 - alpha) * out[sel] + alpha * tint
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(path)
    return path


def save_mask(mask, path):
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)
    return path
