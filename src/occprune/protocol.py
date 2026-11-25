"""Experimental-protocol helpers shared by every pruning study (prune_run, calibration_ablation):
integrity checks on the split, checkpoint and fixed occlusions, the calibration draw, preprocessing
of calibration views and parameter / sparsity accounting."""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from .data import casia_split_path
from .occlusion import sample_train_boxes
from .pruning import prunable_layers
from .transforms import preprocess, to_tensor_images


def array_sha256(array) -> str:
    """SHA-256 of an array's raw bytes (C order)."""
    return hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fixed_conditions(path, expected):
    """Check the published rectangles against the selected protocol and use the saved arrays."""
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(expected) - {"clean"}:
            raise ValueError(f"Unexpected evaluation conditions in {path}")
        result = {"clean": None}
        for name, rects in expected.items():
            if rects is not None:
                value = saved[name]
                if not np.array_equal(value, rects):
                    raise ValueError(f"{name} rectangles disagree with checkpoint protocol in {path}")
                result[name] = value
    return result


def verify_checkpoint_split(path, meta):
    seed = int(meta["seed"])
    digest = file_sha256(path)
    with np.load(path, allow_pickle=False) as z:
        split = dict(z)
    if int(split["seed"]) != seed:
        raise ValueError("Split seed differs from checkpoint metadata")
    if "split_sha256" in meta:
        if digest != meta["split_sha256"]:
            raise ValueError("Split hash differs from checkpoint metadata")
    else:
        # Earlier team checkpoints identify their training split only by seed.
        canonical = casia_split_path(seed)
        if canonical.is_file() and path.resolve() != canonical.resolve():
            with np.load(canonical, allow_pickle=False) as saved:
                if set(split) != set(saved.files) or any(not np.array_equal(split[k], saved[k]) for k in saved.files):
                    raise ValueError("Override differs from the repository split used by this checkpoint")
    return split, digest


def calibration_draw(train_idx, budget, seed, ratio):
    """Halves A and B of `budget` distinct originals plus occlusion boxes for both halves.
    The draw of (A, B, boxes_B) is unchanged from the first team run; boxes_A use their own stream."""
    if budget < 2 or budget % 2 or budget > len(train_idx):
        raise ValueError("Budget must be positive, even and no greater than the training image count")
    rng = np.random.default_rng(seed)
    first, second = np.split(rng.choice(train_idx, size=budget, replace=False), 2)
    boxes_second = sample_train_boxes(len(second), 1.0, tuple(ratio), 112, 112, rng)
    boxes_first = sample_train_boxes(len(first), 1.0, tuple(ratio), 112, 112, np.random.default_rng([seed, 1]))
    return first, second, boxes_first, boxes_second


def scoring_views(images, labels, indices, boxes, fill, dev):
    # Batch preprocessing for I/O locality. taylor_scores differentiates one example at a time.
    for start in range(0, len(indices), 16):
        idx = indices[start:start + 16]
        rects = None if boxes is None else boxes[start:start + len(idx)]
        x = preprocess(to_tensor_images(images[idx], dev), rects, fill)
        yield x, torch.as_tensor(labels[idx], device=dev)


def parameter_counts(model, head, masks=None):
    layers = prunable_layers(model)
    per_layer = {}
    for name, layer in layers.items():
        w = layer.weight.detach()
        if masks is not None and torch.count_nonzero(w.masked_select(~masks[name])).item():
            raise RuntimeError(f"Masked weights in {name} are no longer zero")
        per_layer[name] = {"total": w.numel(), "nonzero": int(torch.count_nonzero(w).item())}
        if masks is not None:
            per_layer[name]["kept_by_mask"] = int(masks[name].sum().item())
    return {
        "backbone_total": sum(p.numel() for p in model.parameters()),
        "backbone_nonzero": sum(int(torch.count_nonzero(p).item()) for p in model.parameters()),
        "prunable_total": sum(v["total"] for v in per_layer.values()),
        "prunable_nonzero": sum(v["nonzero"] for v in per_layer.values()),
        "cosface_head_total": sum(p.numel() for p in head.parameters()),
        "per_layer": per_layer,
    }
