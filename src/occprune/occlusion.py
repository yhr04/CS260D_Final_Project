"""Synthetic square occlusion, represented as integer boxes (y0, x0, side).

x_occ = T_M(x): pixels inside the box are set to the fill value (black) in
[0,1] pixel space, before ImageNet normalisation. The area ratio refers to the
whole image area, not the visible face area.

This module is NumPy-only (training-box sampling); applying boxes to tensors
lives in transforms.py, fixed evaluation occlusions of test pairs in faceverif.py.
"""
from __future__ import annotations

import math

import numpy as np


def patch_side(ratio: float, H: int, W: int) -> int:
    """Side length of a square patch covering ~`ratio` of an HxW image."""
    if ratio <= 0:
        return 0
    return int(min(round(math.sqrt(ratio * H * W)), H, W))


def _place(sides: np.ndarray, H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly place squares fully inside the image. Returns (n,3) int32 [y0,x0,side]."""
    sides = np.asarray(sides, dtype=np.int64)
    y0 = np.floor(rng.random(len(sides)) * (H - sides + 1)).astype(np.int64)
    x0 = np.floor(rng.random(len(sides)) * (W - sides + 1)).astype(np.int64)
    return np.stack([y0, x0, sides], axis=1).astype(np.int32)


def sample_train_boxes(
    n: int,
    prob: float,
    ratio_range: tuple[float, float],
    H: int,
    W: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Training-time occlusion: with prob `prob` an image gets a square whose
    area ratio ~ U(ratio_range); otherwise side = 0 (clean image)."""
    lo, hi = ratio_range
    apply = rng.random(n) < prob
    ratios = rng.uniform(lo, hi, size=n)
    sides = np.minimum(np.rint(np.sqrt(ratios * H * W)), min(H, W)).astype(np.int64)
    sides = np.where(apply, sides, 0)
    return _place(sides, H, W, rng)


def boxes_to_mask_np(boxes: np.ndarray, H: int, W: int) -> np.ndarray:
    """(n,3) boxes -> (n,H,W) bool masks (True = occluded)."""
    y0, x0, s = boxes[:, 0, None, None], boxes[:, 1, None, None], boxes[:, 2, None, None]
    ys = np.arange(H)[None, :, None]
    xs = np.arange(W)[None, None, :]
    return (ys >= y0) & (ys < y0 + s) & (xs >= x0) & (xs < x0 + s)


def apply_boxes_np(images_u8: np.ndarray, boxes: np.ndarray, fill: int = 0) -> np.ndarray:
    """For visualisation / tests: occlude (n,H,W,3) uint8 images (copy)."""
    out = images_u8.copy()
    m = boxes_to_mask_np(boxes, images_u8.shape[1], images_u8.shape[2])
    out[m] = fill
    return out
