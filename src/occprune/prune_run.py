"""One-shot pruning comparison on one Dense-Aug verification checkpoint.

Every method starts from the same checkpoint, uses the same calibration originals and the same
evaluation pairs / fixed occlusions, and is pruned layerwise (unstructured, equal sparsity per layer).

Methods (scoring data per method; B = budget):
    random        uniform random scores (seeded by the scoring seed)        0 backward passes
    magnitude     |w|                                                        0
    clean_taylor  per-image Taylor on B clean views                          B
    occ_taylor    per-image Taylor on B occluded views                       B
    mixed_taylor  per-image Taylor on B/2 clean + B/2 occluded views         B
    oagp          per-layer normalised clean / occluded banks, weighted mean B (same views as mixed)
    oagp_max      per-layer normalised clean / occluded banks, coordinatewise max

Calibration: B originals drawn from the training identities, split into halves A and B.
Four banks of per-image Taylor scores are computed once and shared:
    clean_A, clean_B, occ_A, occ_B   (occluded views use random squares, area ~ U(train_ratio)).
clean_taylor = mean(clean_A, clean_B), occ_taylor = mean(occ_A, occ_B),
mixed_taylor = mean(clean_A, occ_B); oagp / oagp_max combine clean_A and occ_B.
Mixed, OAGP and OAGP-max therefore see exactly the same 1024 views; they differ only in how the
two conditions are combined. Halves A-clean / B-occluded are the draws of the first team run.

Results land in a NEW output directory: result.json, calibration.npz, pair_similarities.npz and,
unless disabled, masks.npz and condition_scores.npz (large, git-ignored).
"""
from __future__ import annotations

import argparse
import copy
import tempfile
from pathlib import Path

import numpy as np
import torch

from .data import casia_split_path, check_casia_split, load_casia, load_pairs, train_indices_and_labels
from .faceverif import dataset_seed, drop_per_pair, embed_kwargs, evaluate_pairs, pair_conditions
from .occlusion import sample_train_boxes
from .pipeline import load_checkpoint
from .protocol import (
    calibration_draw, file_sha256, load_fixed_conditions, parameter_counts, scoring_views,
    verify_checkpoint_split,
)
from .pruning import (
    apply_masks_, condition_scores, enforce_masks_, layerwise_masks, magnitude_scores,
    mean_banks, random_scores, taylor_scores,
)
from .training import epoch_batches, lr_lambda, make_loader, param_groups, train_step
from .utils import Timer, env_info, get_device, load_config, resolve, save_json, set_seed

METHODS = ("random", "magnitude", "clean_taylor", "occ_taylor", "mixed_taylor", "oagp", "oagp_max")
METHOD_LABEL = {"random": "Random", "magnitude": "Magnitude", "clean_taylor": "Clean-Taylor",
                "occ_taylor": "Occ-Taylor", "mixed_taylor": "Mixed-Taylor", "oagp": "OAGP", "oagp_max": "OAGP-max"}
BANKS_USED = {"random": (), "magnitude": (), "clean_taylor": ("clean_A", "clean_B"),
              "occ_taylor": ("occ_A", "occ_B"), "mixed_taylor": ("clean_A", "occ_B"),
              "oagp": ("clean_A", "occ_B"), "oagp_max": ("clean_A", "occ_B")}


def sparsity_key(s: float) -> str:
    return f"s{float(s):g}"


def _method_scores(method, model, banks, alpha, scoring_seed):
    if method == "random":
        return random_scores(model, scoring_seed)
    if method == "magnitude":
        return magnitude_scores(model)
    if method in ("clean_taylor", "occ_taylor", "mixed_taylor"):
        return mean_banks(*(banks[b] for b in BANKS_USED[method]))
    agg = "alpha_weighted" if method == "oagp" else "condition_max"
    return condition_scores(banks["clean_A"], banks["occ_B"], method=agg, alpha=alpha)


def _fine_tune(model, head, masks, labels, train_idx, cfg, seed, epochs, lr, warmup_steps, dev):
    """Augmented fine-tuning (Dense-Aug recipe: random squares w.p. train_prob, flips) with its own
    learning rate: linear warm-up for `warmup_steps`, then cosine from `lr` to 0. Masks are
    re-applied after every optimizer step. Batches and boxes depend only on `seed`, so every
    method sees the same fine-tuning data."""
    t, occ = cfg["train"], cfg["occlusion"]
    bs = t["batch_size"]
    steps = len(train_idx) // bs
    if not steps or bs < 2:
        raise ValueError("Fine-tuning requires a complete batch of at least two training images")
    set_seed(seed, t["deterministic"])
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng([seed, 7])
    opt = torch.optim.SGD(param_groups(model, head, t["weight_decay"]), lr=lr, momentum=t["momentum"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(epochs * steps, warmup_steps))
    model.train()
    head.train()
    lrs = []
    with Timer(dev) as timer:
        for _ in range(epochs):
            batches = epoch_batches(train_idx, bs, g)
            for idx, xb in zip(batches, make_loader(cfg, batches, pin_memory=dev.type == "cuda")):
                boxes = sample_train_boxes(bs, occ["train_prob"], tuple(occ["train_ratio"]), 112, 112, rng)
                flip = torch.rand(bs, generator=g) < 0.5 if t["hflip"] else None
                x8 = xb.to(dev, non_blocking=True).permute(0, 3, 1, 2)
                y = torch.as_tensor(labels[idx], device=dev)
                lrs.append(opt.param_groups[0]["lr"])
                loss, _ = train_step(model, head, opt, sched, x8, y, boxes, flip, cfg)
                enforce_masks_(model, masks)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Non-finite fine-tuning loss")
    model.eval()
    head.eval()
    return {"steps": steps * epochs, "examples": steps * epochs * bs, "seconds": timer.elapsed,
            "lr": lr, "warmup_steps": warmup_steps, "lr_first": lrs[0], "lr_max": max(lrs), "lr_last": lrs[-1]}


def run_pruning(checkpoint, output, *, methods=METHODS, sparsities=(0.5,), cache_dir=None, split_path=None,
                masks_dir="results/masks", data_label="CASIA-WebFace", alpha=None, budget=None,
                scoring_seed=42, ft_epochs=None, ft_lr=None, ft_warmup_steps=None, device="auto",
                eval_batch_size=None, test_sets=(), save_banks=True, save_masks=True):
    """Run every (method, sparsity) on validation (+ `test_sets` once settings are fixed).

    Defaults for alpha / budget / ft_* come from the `prune` section of configs/default.yaml.
    Scores are computed once per method and reused for every sparsity.
    """
    pcfg = load_config().get("prune", {})
    alpha = pcfg.get("alpha", 0.5) if alpha is None else alpha
    budget = pcfg.get("budget", 1024) if budget is None else budget
    ft_epochs = pcfg.get("ft_epochs", 0) if ft_epochs is None else ft_epochs
    ft_lr = pcfg.get("ft_lr", 0.01) if ft_lr is None else ft_lr
    ft_warmup_steps = pcfg.get("ft_warmup_steps", 0) if ft_warmup_steps is None else ft_warmup_steps
    methods = tuple(methods)
    sparsities = tuple(float(s) for s in sparsities)
    out, checkpoint = resolve(output), resolve(checkpoint)
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected Dense-Aug checkpoint with model + CosFace head required: {checkpoint}")
    if not methods or set(methods) - set(METHODS) or len(set(methods)) != len(methods):
        raise ValueError(f"methods must be distinct names from {METHODS}")
    if not sparsities or any(not 0 <= s <= 1 for s in sparsities):
        raise ValueError("sparsities must lie in [0,1]")
    if not 0 <= alpha <= 1 or ft_epochs < 0 or ft_lr <= 0 or ft_warmup_steps < 0:
        raise ValueError("α must be in [0,1]; ft_epochs / ft_warmup_steps nonnegative; ft_lr positive")
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
    cfg["data"]["cache_dir"] = str(cache)          # fine-tuning DataLoader reads the same cache
    split_path = resolve(split_path or casia_split_path(seed))
    images, labels = load_casia(cache)
    if len(labels) != len(images) or images.shape[1:] != (112, 112, 3) or images.dtype != np.uint8:
        raise ValueError("Expected aligned uint8 images and matching labels")
    split, split_hash = verify_checkpoint_split(split_path, meta)
    check_casia_split(split, labels)
    train_idx, remapped, n_cls = train_indices_and_labels(labels, split["holdout_ids"])
    if n_cls != meta["n_classes"]:
        raise ValueError("Training label remapping does not match the CosFace checkpoint")
    occ = cfg["occlusion"]
    first, second, boxes_first, boxes_second = calibration_draw(train_idx, budget, scoring_seed, occ["train_ratio"])
    val_pairs, val_same, val_fold = split["val_pairs"], split["val_issame"], split["fold"]
    if len(np.unique(val_fold)) < 2:
        raise ValueError("Validation requires multiple held-out folds")
    mask_dir = resolve(masks_dir)
    val_expected = pair_conditions(len(val_pairs), occ["val_ratios"], [], dataset_seed(occ["val_seed"], f"casia{seed}"))
    val_mask_path = mask_dir / f"val_seed{seed}_pair_occlusions.npz"
    inputs = {"validation": (images, val_pairs, val_same, val_fold, load_fixed_conditions(val_mask_path, val_expected))}
    mask_hashes = {"validation": file_sha256(val_mask_path)}
    for name in test_sets:
        test = load_pairs(cache, name)
        same = test["issame"]
        pairs = np.arange(len(same) * 2).reshape(-1, 2)
        expected = pair_conditions(len(same), occ["test_ratios"], occ["test_regions"], dataset_seed(occ["test_seed"], name))
        path = mask_dir / f"{name}_pair_occlusions.npz"
        folds = np.arange(len(same)) * 10 // len(same)
        inputs[name] = (test["images"], pairs, same, folds, load_fixed_conditions(path, expected))
        mask_hashes[name] = file_sha256(path)

    result = {
        "data_label": data_label, "checkpoint": checkpoint.name, "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_metadata": meta, "config": cfg, "device": str(dev), "environment": env_info(),
        "split_sha256": split_hash, "eval_mask_sha256": mask_hashes,
        "n_train_images": len(train_idx), "n_train_classes": n_cls,
        "scoring_seed": scoring_seed, "fine_tuning_seed": seed,
        "methods_run": list(methods), "sparsities": list(sparsities), "alpha": alpha, "budget": budget,
        "fine_tuning": {"epochs": ft_epochs, "lr": ft_lr, "warmup_steps": ft_warmup_steps,
                        "schedule": "linear warm-up then per-step cosine to 0; masks re-applied after every step"},
        "calibration": {"originals": budget, "half_size": budget // 2,
                        "estimator": "per-example absolute CosFace Taylor, BN in eval mode, fp32",
                        "normalization": "OAGP: per condition and layer, divide by mean + 1e-12",
                        "banks": {"clean_A": "half A clean", "clean_B": "half B clean",
                                  "occ_A": "half A occluded", "occ_B": "half B occluded"},
                        "bank_seconds": {}},
        "methods": {},
    }
    sims = {}
    for name, (_, _, same, folds, _) in inputs.items():
        sims[f"{name}/issame"], sims[f"{name}/fold"] = same, folds

    def evaluate(model, head, masks, prefix):
        entry = {"parameters": parameter_counts(model, head, masks), "metrics": {}}
        for name, (data, pairs, same, folds, conditions) in inputs.items():
            evaluation = evaluate_pairs(model, data, pairs, same, conditions, folds, **embed_kwargs(cfg, dev))
            if not all(np.isfinite(values).all() for values in evaluation["_sims"].values()):
                raise FloatingPointError(f"Non-finite pair similarities: {prefix}/{name}")
            entry["metrics"][name] = drop_per_pair(evaluation)
            for cond, values in evaluation["_sims"].items():
                sims[f"{prefix}/{name}/{cond}"] = values
        return entry

    result["dense"] = evaluate(dense, dense_head, None, "dense")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pruning-", dir=out.parent) as tmp:
        # Publishing the completed directory makes an interrupted run safe to retry.
        stage = Path(tmp) / "completed"
        stage.mkdir()
        np.savez_compressed(stage / "calibration.npz", half_A_indices=first, half_B_indices=second,
                            half_A_boxes=boxes_first, half_B_boxes=boxes_second)

        # ---- the four shared Taylor banks, only those some selected method needs
        needed = {b for m in methods for b in BANKS_USED[m]}
        views = {"clean_A": (first, None), "clean_B": (second, None),
                 "occ_A": (first, boxes_first), "occ_B": (second, boxes_second)}
        banks = {}
        scoring_model = copy.deepcopy(dense).to(memory_format=torch.contiguous_format)
        for b in ("clean_A", "occ_B", "clean_B", "occ_A"):
            if b in needed:
                idx, bx = views[b]
                with Timer(dev) as tb:
                    banks[b] = taylor_scores(scoring_model, dense_head, scoring_views(images, remapped, idx, bx, occ["fill"], dev))
                result["calibration"]["bank_seconds"][b] = tb.elapsed
        del scoring_model
        if save_banks and banks:
            # fp16 relative to the per-layer mean (avoids fp16 underflow), scale kept in fp32.
            arrays = {}
            for b, bank in banks.items():
                for layer, s in bank.items():
                    scale = float(s.mean()) or 1.0
                    arrays[f"{b}/{layer}"] = (s / scale).numpy().astype(np.float16)
                    arrays[f"scale/{b}/{layer}"] = np.float32(scale)
            np.savez_compressed(stage / "condition_scores.npz", **arrays)

        all_masks = {}
        for method in methods:
            with Timer(dev) as scoring:
                scores = _method_scores(method, dense, banks, alpha, scoring_seed)
            used = BANKS_USED[method]
            m_entry = {"label": METHOD_LABEL[method], "banks": list(used),
                       "backward_examples": sum(budget // 2 for _ in used),
                       "aggregation_seconds": scoring.elapsed,
                       "scoring_seconds": scoring.elapsed + sum(result["calibration"]["bank_seconds"][b] for b in used),
                       "sparsities": {}}
            for sp in sparsities:
                key = sparsity_key(sp)
                model, head = copy.deepcopy(dense), copy.deepcopy(dense_head)
                with Timer(dev) as masking:
                    masks = layerwise_masks(scores, sp)
                    device_masks = apply_masks_(model, masks)
                if save_masks:
                    all_masks.update({f"{method}/{key}/{k}": np.packbits(v.cpu().numpy()) for k, v in masks.items()})
                entry = {"masking_seconds": masking.elapsed,
                         "one_shot": evaluate(model, head, device_masks, f"{method}/{key}/one_shot")}
                if ft_epochs:
                    ft_cfg = cfg
                    if dev.type == "mps":
                        # MPS convolution backward needs contiguous weights and inputs.
                        ft_cfg = {**cfg, "train": {**cfg["train"], "channels_last": False}}
                        model.to(memory_format=torch.contiguous_format)
                    entry["fine_tuning"] = _fine_tune(model, head, device_masks, remapped, train_idx, ft_cfg,
                                                      seed, ft_epochs, ft_lr, ft_warmup_steps, dev)
                    if cfg["train"]["channels_last"]:
                        model.to(memory_format=torch.channels_last)
                    entry["fine_tuned"] = evaluate(model, head, device_masks, f"{method}/{key}/fine_tuned")
                m_entry["sparsities"][key] = entry
                del model, head, masks, device_masks
            result["methods"][method] = m_entry
            del scores
        if save_masks:
            np.savez_compressed(stage / "masks.npz", **all_masks)
        np.savez_compressed(stage / "pair_similarities.npz", **sims)
        save_json(result, stage / "result.json")
        stage.rename(out)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="New directory for results")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--sparsity", dest="sparsities", type=float, nargs="+", default=[0.5])
    parser.add_argument("--cache-dir")
    parser.add_argument("--split-path")
    parser.add_argument("--masks-dir", default="results/masks")
    parser.add_argument("--data-label", default="CASIA-WebFace")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--scoring-seed", type=int, default=42)
    parser.add_argument("--ft-epochs", type=int)
    parser.add_argument("--ft-lr", type=float)
    parser.add_argument("--ft-warmup-steps", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--test-set", dest="test_sets", action="append", default=[],
                        help="Optional final benchmark after validation settings are fixed (e.g. lfw)")
    parser.add_argument("--no-save-banks", dest="save_banks", action="store_false")
    parser.add_argument("--no-save-masks", dest="save_masks", action="store_false")
    args = parser.parse_args()
    run_pruning(**vars(args))
    print(f"Saved {resolve(args.output) / 'result.json'}")


if __name__ == "__main__":
    main()
