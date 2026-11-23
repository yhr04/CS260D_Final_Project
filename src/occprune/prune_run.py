"""Compare magnitude, α-weighted and Condition-Max on one Dense-Aug verification checkpoint."""
from __future__ import annotations

import argparse
import copy
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import torch

from .data import casia_split_path, check_casia_split, load_casia, load_pairs, train_indices_and_labels
from .faceverif import dataset_seed, evaluate_pairs, pair_conditions
from .occlusion import sample_train_boxes
from .pipeline import _embed_kw, _epoch_batches, _lr_lambda, _param_groups, _strip, _train_step, load_checkpoint
from .pruning import (
    condition_scores, apply_masks_, enforce_masks_, layerwise_masks,
    magnitude_scores, prunable_layers, taylor_scores,
)
from .transforms import preprocess, to_tensor_images
from .utils import Timer, env_info, get_device, resolve, save_json, set_seed


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixed_conditions(path, expected):
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


def _checkpoint_split(path, meta):
    seed = int(meta["seed"])
    digest = _file_hash(path)
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


def _calibration(train_idx, budget, seed, ratio):
    if budget < 2 or budget % 2 or budget > len(train_idx):
        raise ValueError("Budget must be positive, even and no greater than the training image count")
    rng = np.random.default_rng(seed)
    clean, occluded = np.split(rng.choice(train_idx, size=budget, replace=False), 2)
    boxes = sample_train_boxes(len(occluded), 1.0, tuple(ratio), 112, 112, rng)
    return clean, occluded, boxes


def _views(images, labels, indices, boxes, fill, dev):
    # Batch preprocessing for I/O locality. taylor_scores differentiates one example at a time.
    for start in range(0, len(indices), 16):
        idx = indices[start:start + 16]
        rects = None if boxes is None else boxes[start:start + len(idx)]
        x = preprocess(to_tensor_images(images[idx], dev), rects, fill)
        yield x, torch.as_tensor(labels[idx], device=dev)


def _fine_tune(model, head, masks, images, labels, train_idx, cfg, seed, epochs, dev):
    t, occ = cfg["train"], cfg["occlusion"]
    bs = t["batch_size"]
    steps = len(train_idx) // bs
    if not steps or bs < 2:
        raise ValueError("Fine-tuning requires a complete batch of at least two training images")
    set_seed(seed, t["deterministic"])
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng([seed, 7])
    opt = torch.optim.SGD(_param_groups(model, head, t["weight_decay"]), lr=t["lr"], momentum=t["momentum"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(epochs * steps, int(t["warmup_epochs"] * steps)))
    model.train()
    head.train()
    with Timer(dev) as timer:
        for _ in range(epochs):
            for idx in _epoch_batches(train_idx, bs, g):
                boxes = sample_train_boxes(bs, occ["train_prob"], tuple(occ["train_ratio"]), 112, 112, rng)
                flip = torch.rand(bs, generator=g) < 0.5 if t["hflip"] else None
                x8 = to_tensor_images(images[idx], dev)
                y = torch.as_tensor(labels[idx], device=dev)
                loss, _ = _train_step(model, head, opt, sched, x8, y, boxes, flip, cfg)
                enforce_masks_(model, masks)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Non-finite fine-tuning loss")
    return {"steps": steps * epochs, "examples": steps * epochs * bs, "seconds": timer.elapsed}


def _counts(model, head, masks=None):
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


def run_pruning(checkpoint, output, *, cache_dir=None, split_path=None, masks_dir="results/masks",
                data_label="CASIA-WebFace", alpha=0.5, sparsity=0.5, budget=1024, scoring_seed=42,
                ft_epochs=0, device="auto", eval_batch_size=None, test_sets=()):
    """Run a fixed setting on validation. Supply test_sets only after selecting settings.

    Reads the selected checkpoint (including its CosFace head and recipe), training cache,
    held-out split and fixed masks. New output contains result.json, packed pruning masks,
    calibration indices/boxes and per-pair similarities, labels and folds.
    """
    out, checkpoint = resolve(output), resolve(checkpoint)
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected Dense-Aug checkpoint with model + CosFace head required: {checkpoint}")
    if not 0 <= alpha <= 1 or not 0 <= sparsity <= 1 or ft_epochs < 0:
        raise ValueError("α and sparsity must be in [0,1]; ft_epochs must be nonnegative")
    dev = get_device(device)
    dense, dense_head, meta = load_checkpoint(checkpoint, dev, weights_only=True)
    if meta["mode"] != "aug":
        raise ValueError("The pruning comparison requires the selected Dense-Aug checkpoint")
    cfg, seed = copy.deepcopy(meta["config"]), int(meta["seed"])
    set_seed(seed, cfg["train"]["deterministic"])
    if eval_batch_size is not None:
        if eval_batch_size < 1:
            raise ValueError("eval_batch_size must be positive")
        cfg["train"]["eval_batch_size"] = eval_batch_size
    if not set(test_sets).issubset(cfg["data"]["test_sets"]):
        raise ValueError("Unrecognised test set")
    cache = resolve(cache_dir or cfg["data"]["cache_dir"])
    split_path = resolve(split_path or casia_split_path(seed))
    images, labels = load_casia(cache)
    if len(labels) != len(images) or images.shape[1:] != (112, 112, 3) or images.dtype != np.uint8:
        raise ValueError("Expected aligned uint8 images and matching labels")
    split, split_hash = _checkpoint_split(split_path, meta)
    check_casia_split(split, labels)
    train_idx, remapped, n_cls = train_indices_and_labels(labels, split["holdout_ids"])
    if n_cls != meta["n_classes"]:
        raise ValueError("Training label remapping does not match the CosFace checkpoint")
    occ = cfg["occlusion"]
    clean, occluded, boxes = _calibration(train_idx, budget, scoring_seed, occ["train_ratio"])
    val_pairs, val_same, val_fold = split["val_pairs"], split["val_issame"], split["fold"]
    if len(np.unique(val_fold)) < 2:
        raise ValueError("Validation requires multiple held-out folds")
    mask_dir = resolve(masks_dir)
    val_name = f"val_seed{seed}"
    val_expected = pair_conditions(len(val_pairs), occ["val_ratios"], [], dataset_seed(occ["val_seed"], f"casia{seed}"))
    val_mask_path = mask_dir / f"{val_name}_pair_occlusions.npz"
    val_masks = _fixed_conditions(val_mask_path, val_expected)
    inputs = {"validation": (images, val_pairs, val_same, val_fold, val_masks)}
    mask_hashes = {"validation": _file_hash(val_mask_path)}
    for name in test_sets:
        test = load_pairs(cache, name)
        same = test["issame"]
        pairs = np.arange(len(same) * 2).reshape(-1, 2)
        expected = pair_conditions(len(same), occ["test_ratios"], occ["test_regions"], dataset_seed(occ["test_seed"], name))
        path = mask_dir / f"{name}_pair_occlusions.npz"
        folds = np.arange(len(same)) * 10 // len(same)
        inputs[name] = (test["images"], pairs, same, folds, _fixed_conditions(path, expected))
        mask_hashes[name] = _file_hash(path)

    result = {
        "data_label": data_label, "checkpoint": checkpoint.name, "checkpoint_sha256": _file_hash(checkpoint),
        "checkpoint_metadata": meta, "config": cfg, "device": str(dev), "environment": env_info(),
        "runtime": {"layout": "channels_last" if cfg["train"]["channels_last"] else "contiguous_nchw",
                    "eval_and_ft_amp": cfg["train"]["amp"], "scoring": "fp32 contiguous",
                    "fine_tuning_layout": "contiguous_nchw" if dev.type == "mps" or not cfg["train"]["channels_last"] else "channels_last"},
        "split_sha256": split_hash, "eval_mask_sha256": mask_hashes,
        "n_train_images": len(train_idx), "n_train_classes": n_cls,
        "scoring_seed": scoring_seed, "fine_tuning_seed": seed, "ft_epochs": ft_epochs,
        "alpha": alpha, "sparsity": sparsity, "budget": budget,
        "calibration": {"clean_views": len(clean), "occluded_views": len(occluded),
                        "unique_originals": budget, "estimator": "per-example absolute CosFace Taylor",
                        "normalization": "per condition and layer, divide by mean + 1e-12"},
        "methods": {},
    }
    sims = {}
    for name, (_, _, same, folds, _) in inputs.items():
        sims[f"{name}/issame"], sims[f"{name}/fold"] = same, folds

    def evaluate(model, head, masks, prefix):
        entry = {"parameters": _counts(model, head, masks), "metrics": {}}
        for name, (data, pairs, same, folds, conditions) in inputs.items():
            evaluation = evaluate_pairs(model, data, pairs, same, conditions, folds, **_embed_kw(cfg, dev))
            if not all(np.isfinite(values).all() for values in evaluation["_sims"].values()):
                raise FloatingPointError(f"Non-finite pair similarities: {prefix}/{name}")
            entry["metrics"][name] = _strip(evaluation)
            for cond, values in evaluation["_sims"].items():
                sims[f"{prefix}/{name}/{cond}"] = values
        return entry

    result["methods"]["dense"] = {"scoring_seconds": 0.0, "masking_seconds": 0.0, "backward_examples": 0,
                                  **evaluate(dense, dense_head, None, "dense")}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pruning-", dir=out.parent) as tmp:
        # Publishing the completed directory makes an interrupted run safe to retry.
        stage = Path(tmp) / "completed"
        stage.mkdir()
        np.savez_compressed(stage / "calibration.npz", clean_indices=clean, occluded_indices=occluded, boxes=boxes)
        scoring_model = copy.deepcopy(dense).to(memory_format=torch.contiguous_format)
        with Timer(dev) as shared:
            clean_scores = taylor_scores(scoring_model, dense_head, _views(images, remapped, clean, None, occ["fill"], dev))
            occ_scores = taylor_scores(scoring_model, dense_head, _views(images, remapped, occluded, boxes, occ["fill"], dev))
        del scoring_model
        result["calibration"]["shared_bank_seconds"] = shared.elapsed
        np.savez_compressed(stage / "condition_scores.npz",
                            **{f"{condition}/{layer}": score.numpy() for condition, bank in
                               (("clean", clean_scores), ("occluded", occ_scores)) for layer, score in bank.items()})
        for method in ("magnitude", "alpha_weighted", "condition_max"):
            model, head = copy.deepcopy(dense), copy.deepcopy(dense_head)
            with Timer(dev) as scoring:
                if method == "magnitude":
                    scores = magnitude_scores(model)
                else:
                    scores = condition_scores(clean_scores, occ_scores, method=method, alpha=alpha)
            with Timer(dev) as masking:
                masks = layerwise_masks(scores, sparsity)
                device_masks = apply_masks_(model, masks)
            del scores
            np.savez_compressed(stage / f"{method}_masks.npz",
                                **{k: np.packbits(v.cpu().numpy()) for k, v in masks.items()})
            entry = {"aggregation_or_magnitude_seconds": scoring.elapsed,
                     "scoring_seconds": scoring.elapsed + (0.0 if method == "magnitude" else shared.elapsed),
                     "masking_seconds": masking.elapsed,
                     "backward_examples": 0 if method == "magnitude" else budget,
                     "one_shot": evaluate(model, head, device_masks, f"{method}/one_shot")}
            if ft_epochs:
                ft_cfg = cfg
                if dev.type == "mps":
                    # MPS convolution backward needs contiguous weights and inputs.
                    ft_cfg = {**cfg, "train": {**cfg["train"], "channels_last": False}}
                    model.to(memory_format=torch.contiguous_format)
                entry["fine_tuning"] = _fine_tune(model, head, device_masks, images, remapped, train_idx,
                                                  ft_cfg, seed, ft_epochs, dev)
                if cfg["train"]["channels_last"]:
                    model.to(memory_format=torch.channels_last)
                entry["fine_tuned"] = evaluate(model, head, device_masks, f"{method}/fine_tuned")
            result["methods"][method] = entry
            del model, head, masks, device_masks
        np.savez_compressed(stage / "pair_similarities.npz", **sims)
        save_json(result, stage / "result.json")
        stage.rename(out)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="New directory for results")
    parser.add_argument("--cache-dir")
    parser.add_argument("--split-path")
    parser.add_argument("--masks-dir", default="results/masks")
    parser.add_argument("--data-label", default="CASIA-WebFace")
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--scoring-seed", type=int, default=42)
    parser.add_argument("--ft-epochs", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--test-set", dest="test_sets", action="append", default=[],
                        help="Optional final benchmark after validation settings are fixed (e.g. lfw)")
    args = parser.parse_args()
    run_pruning(**vars(args))
    print(f"Saved {resolve(args.output) / 'result.json'}")


if __name__ == "__main__":
    main()
