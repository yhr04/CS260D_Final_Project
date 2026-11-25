"""Reproduce the calibration-selection study with the released Dense-Aug checkpoint."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import prune_run as pr
from .calibration_selection import cosine_kernel, select_calibration
from .faceverif import embed
from .pruning import apply_masks_, condition_scores, layerwise_masks, taylor_scores
from .utils import Timer, resolve

STUDY_DIR = "results/pruning/calibration_selection"
SEEDS = (42, 43, 44)
BUDGETS = (256, 512, 1024)
STRATEGIES = ("random", "coreset", "kcenter", "greedi", "greedy_merge", "sieve")
REGIONS = ("eyes", "mouth_nose", "upper_half", "lower_half")
LFW_IMAGES_SHA256 = "5300beeeb04a8244a5da7ceef627d7406f5c9a0fc303ee3ffa02b6a3292d07c0"
LFW_LABELS_SHA256 = "c37b8bcc4e73e0f4cb60cd31df08e54fa687e981ecd23ed5f129f71c41b3c939"


def _check_hash(actual, expected, description):
    if actual != expected:
        raise ValueError(f"{description} differs from the recorded experiment: {actual} (expected {expected})")


def _array_hash(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()


def _training_pool(cache_dir, meta, protocol):
    """Recreate the union of the original 1024-image training draws, including at smaller B."""
    split_path = resolve("results/splits/split_seed42.npz")
    split, digest = pr._checkpoint_split(split_path, meta)
    _check_hash(digest, protocol["split_sha256"], "CASIA training split")
    images, labels = pr.load_casia(cache_dir)
    pr.check_casia_split(split, labels)
    train, remapped, n_classes = pr.train_indices_and_labels(labels, split["holdout_ids"])
    if (len(train), n_classes) != (481210, meta["n_classes"]):
        raise ValueError("Unexpected CASIA training population or CosFace class remapping")

    draws = {}
    for seed in SEEDS:
        first, second, _, boxes = pr._calibration(train, 1024, seed, meta["config"]["occlusion"]["train_ratio"])
        draws[seed] = np.column_stack((first, second)).ravel(), boxes
    ids = np.unique(np.concatenate([order for order, _ in draws.values()]))
    candidates, targets = images[ids], remapped[ids]
    _check_hash(_array_hash(ids.astype("<i8")), protocol["candidate_pool_sha256"], "Sorted CASIA photo IDs")
    _check_hash(_array_hash(targets.astype("<i8")), protocol["labels_sha256"], "Remapped CASIA targets")
    _check_hash(_array_hash(candidates), protocol["pixels_sha256"], "Decoded training pixels")
    return ids, candidates, targets, draws


def load_lfw(cache_dir, cfg, protocol):
    """Return the eight fixed test conditions, the 6,000 original pairs and their ten folds."""
    data = pr.load_pairs(cache_dir, "lfw")
    images, same = data["images"], data["issame"]
    _check_hash(_array_hash(images), LFW_IMAGES_SHA256, "Decoded LFW pixels")
    _check_hash(_array_hash(same), LFW_LABELS_SHA256, "LFW pair labels")
    occ = cfg["occlusion"]
    expected = pr.pair_conditions(len(same), occ["test_ratios"], occ["test_regions"],
                                  pr.dataset_seed(occ["test_seed"], "lfw"))
    path = resolve("results/masks/lfw_pair_occlusions.npz")
    _check_hash(pr._file_hash(path), protocol["lfw_masks_sha256"], "LFW rectangles")
    conditions = pr._fixed_conditions(path, expected)
    pairs = np.arange(2 * len(same)).reshape(-1, 2)
    folds = np.arange(len(same)) * 10 // len(same)
    return images, pairs, same, folds, conditions


def score_and_evaluate(dense, head, cfg, device, candidates, targets, order, boxes, benchmark):
    """Score alternating clean/occluded ranks once; apply each pruning rule to a fresh dense copy."""
    order = np.asarray(order)
    if (order.ndim != 1 or order.dtype.kind not in "iu" or not len(order) or len(order) % 2
            or np.unique(order).size != len(order) or order.min() < 0 or order.max() >= len(candidates)
            or np.shape(boxes) != (len(order) // 2, 3)):
        raise ValueError("Invalid calibration order or square boxes")

    scorer = copy.deepcopy(dense).to(memory_format=torch.contiguous_format).eval()
    with Timer(device) as backward:
        banks = []
        for rows, rects in ((order[::2], None), (order[1::2], boxes)):
            views = pr._views(candidates, targets, rows, rects, cfg["occlusion"]["fill"], device)
            banks.append(taylor_scores(scorer, head, views))
    del scorer
    images, pairs, same, folds, conditions = benchmark
    embed_kw = {**pr._embed_kw(cfg, device), "batch_size": 64}
    methods = {}
    for name in ("alpha_weighted", "condition_max"):
        with Timer(device) as aggregation:
            scores = condition_scores(*banks, method=name, alpha=0.5)
        with Timer(device) as masking:
            masks = layerwise_masks(scores, 0.5)
        model = copy.deepcopy(dense).eval()
        device_masks = apply_masks_(model, masks)
        counts = pr._counts(model, head, device_masks)
        if (len(masks), counts["prunable_nonzero"], counts["prunable_total"]) != (19, 5578752, 11157504):
            raise ValueError("Unexpected sparsity or prunable layers")
        with Timer(device) as evaluation:
            result = pr.evaluate_pairs(model, images, pairs, same, conditions, folds, **embed_kw)
        if any(not np.isfinite(s).all() for s in result["_sims"].values()):
            raise FloatingPointError("Non-finite LFW similarity")
        metrics = pr._strip(result)
        methods[name] = {
            "metrics": metrics, "parameters": counts, "region_mean_pct": 100 * np.mean([metrics[c]["acc"] for c in REGIONS]),
            "aggregation_seconds": aggregation.elapsed, "mask_seconds": masking.elapsed,
            "evaluation_seconds": evaluation.elapsed,
        }
    return {"methods": methods, "shared_backward_seconds": backward.elapsed}


def rerun_calibration(*, checkpoint="checkpoints/dense_aug_seed42.pt", cache_dir="data/cache",
                      seeds=(42,), budgets=(256,), strategies=("sieve",), selection_mode="replay", device="auto"):
    """Yield one result per selector/seed/budget. Read recorded orders or rebuild selectors on this device."""
    if selection_mode not in ("replay", "recompute"):
        raise ValueError("selection_mode must be replay or recompute")
    for name, subset, allowed in (("seeds", seeds, SEEDS), ("budgets", budgets, BUDGETS),
                                 ("strategies", strategies, STRATEGIES)):
        if not subset or len(set(subset)) != len(subset) or not set(subset) <= set(allowed):
            raise ValueError(f"{name} must contain distinct choices from {allowed}")

    study = resolve(STUDY_DIR)
    protocol = json.loads((study / "protocol.json").read_text(encoding="utf-8"))
    archive = study / "selection_orders.npz"
    _check_hash(pr._file_hash(archive), protocol["selection_orders_sha256"], "Recorded selection orders")
    checkpoint, cache = resolve(checkpoint), resolve(cache_dir)
    _check_hash(pr._file_hash(checkpoint), protocol["checkpoint_sha256"], "Dense-Aug checkpoint")
    dev = pr.get_device(device)
    dense, head, meta = pr.load_checkpoint(checkpoint, dev, weights_only=True)
    dense.eval()
    head.eval()
    cfg = meta["config"]
    pr.set_seed(meta["seed"], cfg["train"]["deterministic"])
    ids, candidates, targets, draws = _training_pool(cache, meta, protocol)
    benchmark = load_lfw(cache, cfg, protocol)

    kernel = None
    if selection_mode == "recompute":
        features = embed(dense, candidates, device=dev, batch_size=64, amp="off", flip=False, channels_last=True)
        kernel = cosine_kernel(features)
    with np.load(archive, allow_pickle=False) as saved:
        np.testing.assert_array_equal(ids, saved["pool_original_ids"])
        for seed in seeds:
            original_ids, boxes = draws[seed]
            random_order = np.searchsorted(ids, original_ids)
            np.testing.assert_array_equal(random_order, saved[f"random/{seed}/1024"])
            np.testing.assert_array_equal(boxes, saved[f"boxes/{seed}"])
            if selection_mode == "recompute":
                choices = select_calibration(kernel, ids, random_order, seed=seed, budgets=budgets, strategies=strategies)
            for budget in budgets:
                for strategy in strategies:
                    recorded = saved[f"{strategy}/{seed}/{budget}"].astype(np.int64)
                    order = recorded if selection_mode == "replay" else choices[strategy, budget][0]
                    result = score_and_evaluate(dense, head, cfg, dev, candidates, targets, order,
                                                boxes[:budget // 2], benchmark)
                    yield {
                        "strategy": strategy, "seed": seed, "budget": budget, "selection_mode": selection_mode,
                        "device": str(dev), "ordered_ranks_matching_recorded": int(np.sum(order == recorded)),
                        "photos_in_common_with_recorded": int(len(np.intersect1d(order, recorded))), **result,
                    }
