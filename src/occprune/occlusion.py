"""Synthetic square occlusion, represented as integer boxes (y0, x0, side).

x_occ = T_M(x): pixels inside the box are set to the fill value (black) in
[0,1] pixel space, before ImageNet normalisation. The area ratio refers to the
whole image area, not the visible face area.

This module is NumPy-only (box sampling + persistence); applying boxes to
tensors lives in transforms.py.
"""
from __future__ import annotations

import math

import numpy as np

from .utils import resolve


def patch_side(ratio: float, H: int, W: int) -> int:
    """Side length of a square patch covering ~`ratio` of an HxW image."""
    if ratio <= 0:
        return 0
    return int(min(round(math.sqrt(ratio * H * W)), H, W))


def realized_ratio(side: int, H: int, W: int) -> float:
    return side * side / float(H * W)


def _place(sides: np.ndarray, H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly place squares fully inside the image. Returns (n,3) int32 [y0,x0,side]."""
    sides = np.asarray(sides, dtype=np.int64)
    y0 = np.floor(rng.random(len(sides)) * (H - sides + 1)).astype(np.int64)
    x0 = np.floor(rng.random(len(sides)) * (W - sides + 1)).astype(np.int64)
    return np.stack([y0, x0, sides], axis=1).astype(np.int32)


def sample_fixed_ratio_boxes(n: int, ratio: float, H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    side = patch_side(ratio, H, W)
    return _place(np.full(n, side), H, W, rng)


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


def make_eval_boxes(n: int, ratios, repeats: int, H: int, W: int, seed: int) -> dict:
    """Fixed evaluation masks: {ratio: int32 array (R, n, 3)}.

    Each (ratio, repeat) pair uses its own RNG stream derived from `seed`, so
    validation/test masks are independent of each other and of training masks,
    and identical for every method that is evaluated.
    Ratio 0 has a single repeat (no randomness).
    """
    out = {}
    for ratio in ratios:
        R = 1 if ratio <= 0 else repeats
        reps = []
        for r in range(R):
            rng = np.random.default_rng([seed, int(round(ratio * 1000)), r])
            reps.append(sample_fixed_ratio_boxes(n, ratio, H, W, rng))
        out[float(ratio)] = np.stack(reps, axis=0)
    return out


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


# --------------------------------------------------------------------------- persistence
def eval_boxes_path(split_name: str, split_seed: int):
    return resolve(f"results/masks/{split_name}_boxes_split{split_seed}.npz")


def save_eval_boxes(boxes: dict, split_name: str, split_seed: int, meta: dict) -> None:
    p = eval_boxes_path(split_name, split_seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"r{int(round(k * 1000)):04d}": v for k, v in boxes.items()}
    np.savez_compressed(p, **arrays, meta=np.array(repr(meta)))


def load_eval_boxes(split_name: str, split_seed: int) -> dict:
    z = np.load(eval_boxes_path(split_name, split_seed), allow_pickle=False)
    return {int(k[1:]) / 1000.0: z[k] for k in z.files if k.startswith("r")}
