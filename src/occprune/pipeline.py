"""High-level experiment steps, called from the notebooks (the main entry point).

    from occprune.pipeline import prepare_data, run_dense, summarize_dense
    prepare_data()                 # LFW download, split, fixed masks, stats, figures
    run_dense("clean")             # Dense-Clean
    run_dense("aug")               # Dense-Aug (shared start for pruning)
    table, figs = summarize_dense()

Figures and printed logs are in English (they go into the English report);
the notebooks explain each step in Chinese.
"""
from __future__ import annotations

import gc
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import (
    check_split,
    class_counts,
    dataset_fingerprint,
    load_lfw,
    load_split,
    save_split,
    split_path,
    stratified_split,
)
from .occlusion import (
    apply_boxes_np,
    eval_boxes_path,
    load_eval_boxes,
    make_eval_boxes,
    patch_side,
    realized_ratio,
    sample_fixed_ratio_boxes,
    save_eval_boxes,
)
from .utils import Timer, ensure_dir, env_info, get_device, get_logger, load_config, load_json, resolve, save_json, set_seed

COLORS = {"Dense-Clean": "#4C72B0", "Dense-Aug": "#DD8452"}
METHOD_NAME = {"clean": "Dense-Clean", "aug": "Dense-Aug"}


def _cfg_seed(cfg, seed):
    cfg = cfg if cfg is not None else load_config()
    return cfg, (seed if seed is not None else cfg["seed"])


# =============================================================================
# Step 1: data
# =============================================================================
def prepare_data(cfg: dict | None = None, seed: int | None = None, force: bool = False, verbose: bool = True) -> dict:
    """Download LFW (once), make the per-identity split and the fixed val/test
    occlusion masks, save dataset statistics and two figures.

    Idempotent: if the split, masks and stats for this seed already exist they
    are reused (pass force=True to regenerate; the result is deterministic).
    """
    cfg, seed = _cfg_seed(cfg, seed)
    dcfg, ocfg = cfg["data"], cfg["occlusion"]
    stats_path = resolve(f"results/dataset_stats_seed{seed}.json")
    done = all(p.exists() for p in (stats_path, split_path(seed), eval_boxes_path("val", seed), eval_boxes_path("test", seed)))
    if done and not force:
        if verbose:
            print(f"[prepare_data] seed {seed}: split, masks and stats already exist -> reusing (force=True to redo)")
        return load_json(stats_path)

    d = load_lfw(dcfg, verbose=verbose)
    images, labels, names = d["images"], d["labels"], d["target_names"]
    N, H, W, C = images.shape
    counts = class_counts(labels, len(names))
    stats = {
        "n_images": int(N),
        "image_shape_hwc": [int(H), int(W), int(C)],
        "n_identities": int(len(names)),
        "min_faces_per_person": dcfg["min_faces_per_person"],
        "crop_y0y1x0x1": dcfg["crop"],
        "images_per_identity": {"min": int(counts.min()), "median": float(np.median(counts)),
                                "mean": float(counts.mean()), "max": int(counts.max())},
        "largest_class": names[int(counts.argmax())],
        "majority_class_fraction": float(counts.max() / N),
        "pixel_dtype": str(images.dtype),
        "fingerprint": dataset_fingerprint(labels, images),
    }

    # ---------------- split (original images, before any occlusion)
    split = stratified_split(labels, tuple(dcfg["split"]), seed)
    check_split(split, labels)
    save_split(split, labels, seed, dcfg["split"])
    per_split = {k: class_counts(labels[v], len(names)) for k, v in split.items()}
    stats["split_sizes"] = {k: int(len(v)) for k, v in split.items()}
    stats["split_min_per_class"] = {k: int(v.min()) for k, v in per_split.items()}
    res_dir = ensure_dir("results")
    df = pd.DataFrame({"identity": names, "total": counts, **per_split})
    df.sort_values("total", ascending=False).to_csv(res_dir / "class_distribution.csv", index=False)

    # ---------------- fixed evaluation masks (val / test, independent seeds)
    meta = {"H": H, "W": W, "ratios": ocfg["eval_ratios"], "fill": ocfg["fill"]}
    for name, seed_key, rep_key in (("val", "val_seed", "eval_repeats_val"), ("test", "test_seed", "eval_repeats_test")):
        boxes = make_eval_boxes(len(split[name]), ocfg["eval_ratios"], ocfg[rep_key], H, W, ocfg[seed_key] + seed)
        save_eval_boxes(boxes, name, seed, {**meta, "seed": ocfg[seed_key] + seed, "repeats": ocfg[rep_key]})
    stats["occlusion_patch"] = {
        f"{int(r * 100)}%": {"side_px": patch_side(r, H, W), "realized_ratio": round(realized_ratio(patch_side(r, H, W), H, W), 4)}
        for r in ocfg["eval_ratios"]
    }
    stats["env"] = env_info()
    save_json(stats, stats_path)

    # ---------------- figures
    fig_dir = ensure_dir("figures")
    order = np.argsort(-counts)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.bar(range(len(names)), counts[order], color="#4C72B0")
    ax.set_xlabel("identity (sorted by #images)")
    ax.set_ylabel("#images")
    ax.set_title(f"LFW-funneled, min_faces_per_person={dcfg['min_faces_per_person']}: {N} images, {len(names)} identities")
    ax.set_xlim(-0.6, len(names) - 0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "class_distribution.png", dpi=200)
    plt.close(fig)

    rng = np.random.default_rng(0)
    show = rng.choice(split["train"], size=6, replace=False)
    ratios = ocfg["eval_ratios"]
    fig, axes = plt.subplots(len(show), len(ratios), figsize=(1.9 * len(ratios), 1.9 * len(show)))
    for i, idx in enumerate(show):
        for j, r in enumerate(ratios):
            b = sample_fixed_ratio_boxes(1, r, H, W, np.random.default_rng([i, j]))
            axes[i, j].imshow(apply_boxes_np(images[idx: idx + 1], b)[0])
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title("clean" if r == 0 else f"{int(r * 100)}% occluded", fontsize=9)
        axes[i, 0].set_ylabel(names[labels[idx]].split()[-1], fontsize=8)
    fig.suptitle("Synthetic square occlusion (training images)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "occlusion_samples.png", dpi=200)
    plt.close(fig)

    if verbose:
        s = {k: v for k, v in stats.items() if k != "env"}
        print(f"[prepare_data] seed {seed}: {s['n_images']} images, {s['n_identities']} identities, "
              f"split {s['split_sizes']}, min/class {s['split_min_per_class']}")
    return stats


# =============================================================================
# Step 2: dense baselines
# =============================================================================
def dense_name(mode: str, seed: int, tag: str = "") -> str:
    return f"dense_{mode}_seed{seed}" + (f"_{tag}" if tag else "")


def load_dense_result(mode: str, seed: int = 42, tag: str = "") -> dict:
    return load_json(f"results/dense/{dense_name(mode, seed, tag)}.json")


def run_dense(
    mode: str,
    cfg: dict | None = None,
    seed: int | None = None,
    epochs: int | None = None,
    device: str | None = None,
    tag: str = "",
    skip_if_exists: bool = False,
) -> dict:
    """Train Dense-Clean (mode='clean') or Dense-Aug (mode='aug'), select the
    best epoch on validation, evaluate it on the fixed test masks and save
    checkpoint + JSON result + per-image predictions. Returns the result dict.

    Logs are printed live and written to results/logs/<name>.log.
    """
    import torch

    from .evaluate import evaluate_levels, train_set_diagnostics
    from .models import build_resnet18, count_parameters, load_checkpoint, save_checkpoint
    from .train import train_model
    from .transforms import to_tensor_images

    assert mode in METHOD_NAME, f"mode must be one of {list(METHOD_NAME)}"
    cfg, seed = _cfg_seed(cfg, seed)
    cfg = {**cfg, "train": {**cfg["train"]}}
    if epochs is not None:
        cfg["train"]["epochs"] = epochs
    if device is not None:
        cfg["train"]["device"] = device
    name = dense_name(mode, seed, tag)
    out_path = resolve(f"results/dense/{name}.json")
    if skip_if_exists and out_path.exists():
        print(f"[run_dense] {name}: result exists -> loading {out_path.name} (skip_if_exists=False to retrain)")
        return load_json(out_path)
    prepare_data(cfg, seed, verbose=False)

    logger = get_logger(name, f"results/logs/{name}.log")
    set_seed(seed, cfg["train"]["deterministic"])
    dev = get_device(cfg["train"]["device"])
    env = env_info()
    logger.info(f"run {name} | device {dev} | gpu {env.get('gpu')} | torch {env.get('torch')}")

    d = load_lfw(cfg["data"], verbose=False)
    labels_np = d["labels"]
    num_classes = len(d["target_names"])
    split = load_split(seed, labels_np)
    val_boxes = load_eval_boxes("val", seed)
    test_boxes = load_eval_boxes("test", seed)
    images = to_tensor_images(d["images"], dev)
    labels = torch.as_tensor(labels_np, device=dev)
    logger.info(f"data {tuple(d['images'].shape)} classes {num_classes} | train {len(split['train'])} "
                f"val {len(split['val'])} test {len(split['test'])} | input {cfg['data']['img_size']}px")

    set_seed(seed, cfg["train"]["deterministic"])  # identical initial head for clean / aug
    model = build_resnet18(num_classes, pretrained=True).to(dev)
    ckpt_path = resolve(f"checkpoints/{name}.pt")

    def on_new_best(m, ep, rec):
        save_checkpoint(ckpt_path, m, {"name": name, "mode": mode, "seed": seed, "epoch": ep,
                                       "val_acc": rec["val_acc"], "config": cfg})

    t0 = time.time()
    tr = train_model(model, images, labels, split, val_boxes, cfg, mode, seed, logger, on_new_best)
    wall = time.time() - t0
    logger.info(f"training done: best epoch {tr['best_epoch']} (val score {tr['best_val_score']:.4f}), "
                f"pure train time {tr['train_time_s']:.1f}s, wall {wall:.1f}s")

    # ---------- final evaluation of the selected checkpoint
    model, _ = load_checkpoint(ckpt_path, num_classes, dev)
    ocfg, img_size, bs = cfg["occlusion"], cfg["data"]["img_size"], cfg["train"]["eval_batch_size"]
    with Timer(dev) as te:
        test = evaluate_levels(model, images, labels, split["test"], test_boxes, img_size, ocfg["fill"], bs, keep_preds=True)
    val = evaluate_levels(model, images, labels, split["val"], val_boxes, img_size, ocfg["fill"], bs)
    diag = train_set_diagnostics(model, images, labels, split["train"], cfg)
    for part, r in (("val", val), ("test", test)):
        logger.info(f"{part}: " + " | ".join(
            f"{int(k * 100)}%: acc {v['acc']:.4f} bal {v['bal_acc']:.4f}" for k, v in r["levels"].items()))
    logger.info("diagnostics: " + ", ".join(
        f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in diag.items()))

    # per-image predictions, for paired significance tests (McNemar / bootstrap)
    ensure_dir("results/dense")
    np.savez_compressed(
        resolve(f"results/dense/{name}_test_preds.npz"),
        y_true=test["y_true"], test_idx=split["test"],
        **{f"r{int(round(k * 1000)):04d}": v for k, v in test["preds"].items()},
    )
    for r in (test, val):
        r.pop("preds", None)
        r.pop("y_true", None)
    result = {
        "name": name,
        "method": METHOD_NAME[mode],
        "mode": mode,
        "seed": seed,
        "sparsity": 0.0,
        "best_epoch": tr["best_epoch"],
        "best_val_score": tr["best_val_score"],
        "select_metric": cfg["train"]["select_metric"],
        "val": val,
        "test": test,
        "train_diagnostics": diag,
        "params": count_parameters(model),
        "timing": {
            "train_time_s": tr["train_time_s"],
            "wall_time_s": wall,
            "test_eval_time_s": te.elapsed,
            "device": str(dev),
            "note": "train_time_s sums per-epoch training loops (accelerator-synchronised); "
                    "wall_time_s also includes per-epoch validation and checkpointing.",
        },
        "history": tr["history"],
        "checkpoint": str(ckpt_path.relative_to(resolve("."))),
        "config": cfg,
        "env": env,
    }
    save_json(result, out_path)
    logger.info(f"saved {out_path}")

    # free accelerator memory so several runs can share one notebook kernel
    del model, images, labels
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return load_json(out_path)


# =============================================================================
# Step 3: tables and figures
# =============================================================================
def _level_tag(k) -> str:
    return "Clean" if float(k) == 0 else f"Occ {int(float(k) * 100)}%"


def summarize_dense(seed: int = 42, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Collect Dense-Clean / Dense-Aug results into a CSV/Markdown table and
    figures. Returns (table, {figure name: matplotlib Figure})."""
    rows, runs = [], {}
    for mode in METHOD_NAME:
        p = resolve(f"results/dense/{dense_name(mode, seed)}.json")
        if not p.exists():
            print(f"[summarize_dense] missing {p.name}")
            continue
        r = load_json(p)
        runs[r["method"]] = r
        lv = r["test"]["levels"]
        keys = sorted(lv, key=float)
        row = {"Method": r["method"], "Seed": r["seed"], "Sparsity": "0%", "Best epoch": r["best_epoch"]}
        row.update({f"{_level_tag(k)} Acc": round(100 * lv[k]["acc"], 2) for k in keys})
        row.update({f"{_level_tag(k)} BalAcc": round(100 * lv[k]["bal_acc"], 2) for k in keys})
        row.update({f"Drop {int(float(k) * 100)}%": round(100 * v, 2) for k, v in r["test"].get("robustness_drop", {}).items()})
        row["Val mean Acc"] = round(100 * r["val"]["mean_acc"], 2)
        row["Train time (s)"] = round(r["timing"]["train_time_s"], 1)
        row["Device"] = r["env"].get("gpu") or r["timing"]["device"]
        row["Scoring Time"] = "N/A"
        rows.append(row)
    if not rows:
        raise FileNotFoundError("no dense results found - run run_dense('clean') / run_dense('aug') first")

    df = pd.DataFrame(rows)
    out_dir = ensure_dir("results")
    df.to_csv(out_dir / f"baseline_table_seed{seed}.csv", index=False)
    md = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    md += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    (out_dir / f"baseline_table_seed{seed}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    fig_dir = ensure_dir("figures")
    figs = {}
    # ---- accuracy vs occlusion ratio
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, metric, title in ((axes[0], "acc", "accuracy"), (axes[1], "bal_acc", "balanced accuracy")):
        for m, r in runs.items():
            lv = r["test"]["levels"]
            ks = sorted(lv, key=float)
            xs = [100 * float(k) for k in ks]
            ys = [100 * lv[k][metric] for k in ks]
            ax.plot(xs, ys, marker="o", lw=2, color=COLORS[m], label=m)
            dy = 7 if m == "Dense-Aug" else -13  # keep labels from overlapping
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, dy), ha="center",
                            fontsize=8, color=COLORS[m])
        ax.set_xlabel("occluded image area (%)")
        ax.set_title(f"Test {title} (seed {seed})")
        ax.set_xticks([0, 10, 30, 50])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("%")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / f"dense_clean_vs_aug_seed{seed}.png", dpi=200)
    fig.savefig(fig_dir / f"dense_clean_vs_aug_seed{seed}.pdf")
    plt.close(fig)
    figs["accuracy_vs_occlusion"] = fig

    # ---- training curves
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for m, r in runs.items():
        h = r["history"]
        ep = [e["epoch"] for e in h]
        axes[0].plot(ep, [e["train_loss"] for e in h], color=COLORS[m], label=m)
        axes[1].plot(ep, [100 * e["val_acc"]["0.0"] for e in h], color=COLORS[m], label=f"{m} clean")
        axes[1].plot(ep, [100 * e["val_acc"]["0.3"] for e in h], color=COLORS[m], ls="--", label=f"{m} 30%")
        axes[2].plot(ep, [100 * e["val_acc"]["0.5"] for e in h], color=COLORS[m], label=m)
        axes[1].axvline(r["best_epoch"], color=COLORS[m], alpha=0.3, lw=1)
    axes[0].set_title("train loss")
    axes[1].set_title("val acc (solid clean, dashed 30%)")
    axes[2].set_title("val acc, 50% occlusion")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / f"dense_training_curves_seed{seed}.png", dpi=200)
    plt.close(fig)
    figs["training_curves"] = fig

    if verbose:
        print(f"[summarize_dense] table -> results/baseline_table_seed{seed}.csv, figures -> figures/")
    return df, figs


def paired_bootstrap(mode_a: str, mode_b: str, seed: int = 42, n_boot: int = 10000, rng_seed: int = 0) -> pd.DataFrame:
    """Accuracy difference (b - a) per occlusion level with a 95% paired
    bootstrap CI over the original test images (masks averaged per image)."""
    pa = np.load(resolve(f"results/dense/{dense_name(mode_a, seed)}_test_preds.npz"))
    pb = np.load(resolve(f"results/dense/{dense_name(mode_b, seed)}_test_preds.npz"))
    y = pa["y_true"]
    boot = np.random.default_rng(rng_seed).choice(len(y), (n_boot, len(y)))
    rows = []
    for key in sorted(k for k in pa.files if k.startswith("r")):
        diff = (pb[key] == y).mean(0) - (pa[key] == y).mean(0)
        lo, hi = np.percentile(diff[boot].mean(1), [2.5, 97.5])
        rows.append({"occlusion": f"{int(key[1:]) // 10}%", "diff_pp": 100 * diff.mean(),
                     "ci_low_pp": 100 * lo, "ci_high_pp": 100 * hi, "n_images": len(y)})
    return pd.DataFrame(rows).round(2)
