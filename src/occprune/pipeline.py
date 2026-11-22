"""Experiment pipeline (CASIA-WebFace training -> LFW verification): data preparation, dense training
(Dense-Clean / Dense-Aug with CosFace), evaluation on clean and occluded pairs.

Notebook entry points (01_data, 02_dense_baselines):

    from occprune.pipeline import prepare_data, throughput_test, run_dense, summarize_dense
    prepare_data(cfg)                    # caches, held-out split, fixed occlusions, stats, figure
    throughput_test(cfg)                  # images/s of loader and training step
    run_dense("clean", cfg)          # resumable; checkpoints every epoch
    run_dense("aug", cfg)
    table, figs = summarize_dense()

Long runs can also be started from a terminal (same functions):

    python -m occprune.pipeline train --mode aug
"""
from __future__ import annotations

import argparse
import gc
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import (
    build_casia_cache,
    build_pairs_cache,
    casia_paths,
    casia_split_path,
    check_casia_split,
    load_casia,
    load_casia_split,
    load_pairs,
    make_casia_split,
    save_casia_split,
    train_indices_and_labels,
)
from .faceverif import (
    REGION_RECTS,
    CosFaceHead,
    build_face_resnet18,
    dataset_seed,
    evaluate_pairs,
    pair_conditions,
    save_conditions,
)
from .occlusion import sample_train_boxes
from .transforms import preprocess
from .utils import Timer, ensure_dir, env_info, get_device, get_logger, load_config, load_json, resolve, save_json, set_seed

METHOD_NAME = {"clean": "Dense-Clean", "aug": "Dense-Aug"}
COLORS = {"Dense-Clean": "#4C72B0", "Dense-Aug": "#DD8452"}
CONFIG = "configs/default.yaml"


def _cfg_seed(cfg, seed):
    cfg = cfg if cfg is not None else load_config(CONFIG)
    return cfg, (seed if seed is not None else cfg["seed"])


def run_name(mode: str, seed: int, tag: str = "") -> str:
    return f"dense_{mode}_seed{seed}" + (f"_{tag}" if tag else "")


# =============================================================================
# Step 1: data
# =============================================================================
def prepare_data(cfg: dict | None = None, seed: int | None = None, workers: int = 12, verbose: bool = True) -> dict:
    """Decode both archives into caches (once), make the held-out identity split and
    validation pairs, save the fixed test occlusions, dataset statistics and a figure."""
    cfg, seed = _cfg_seed(cfg, seed)
    dcfg, ocfg = cfg["data"], cfg["occlusion"]
    meta = build_casia_cache(dcfg["webface_zip"], dcfg["cache_dir"], workers=workers, verbose=verbose)
    for name in dcfg["test_sets"]:
        build_pairs_cache(dcfg["val_zip"], dcfg["cache_dir"], name)

    images, labels = load_casia(dcfg["cache_dir"])
    if not casia_split_path(seed).exists():
        split = make_casia_split(labels, dcfg["holdout_frac"], dcfg["holdout_min_images"], dcfg["val_pairs_per_class"], seed)
        save_casia_split(split, seed)
    split = load_casia_split(seed)
    check_casia_split(split, labels)
    train_idx, _, n_train_cls = train_indices_and_labels(labels, split["holdout_ids"])

    # fixed occlusions of the 2nd image of every test pair (saved for the record; regenerated identically)
    for name in dcfg["test_sets"]:
        conds = pair_conditions(6000, ocfg["test_ratios"], ocfg["test_regions"], dataset_seed(ocfg["test_seed"], name))
        save_conditions(conds, name)
    val_conds = pair_conditions(len(split["val_pairs"]), ocfg["val_ratios"], [], dataset_seed(ocfg["val_seed"], f"casia{seed}"))
    save_conditions(val_conds, f"val_seed{seed}")

    counts = np.bincount(labels)
    stats = {
        "casia": meta,
        "train_images": int(len(train_idx)),
        "train_identities": int(n_train_cls),
        "holdout_identities": int(len(split["holdout_ids"])),
        "holdout_images": int(np.isin(labels, split["holdout_ids"]).sum()),
        "val_pairs": int(len(split["val_pairs"])),
        "val_positive_pairs": int(split["val_issame"].sum()),
        "identities_with_lt5_images": int((counts < 5).sum()),
        "test_sets": {n: {"pairs": int(len(load_pairs(dcfg["cache_dir"], n)["issame"])),
                          "positive": int(load_pairs(dcfg["cache_dir"], n)["issame"].sum())} for n in dcfg["test_sets"]},
        "region_rects_y0x0hw": REGION_RECTS,
        "seed": seed,
    }
    save_json(stats, f"results/dataset_stats_seed{seed}.json")

    # ---- figure: LFW pairs with every test condition on the second image
    lfw = load_pairs(dcfg["cache_dir"], "lfw")
    conds = pair_conditions(6000, ocfg["test_ratios"], ocfg["test_regions"], dataset_seed(ocfg["test_seed"], "lfw"))
    show = [0, 1, 300, 301, 2400]      # two positive, two negative, one more positive pair
    fig, axes = plt.subplots(len(show), len(conds) + 1, figsize=(1.45 * (len(conds) + 1), 1.55 * len(show)))
    for i, p in enumerate(show):
        axes[i, 0].imshow(lfw["images"][2 * p])
        axes[i, 0].set_ylabel("same" if lfw["issame"][p] else "different", fontsize=8)
        for j, (cn, rects) in enumerate(conds.items(), start=1):
            x = torch.from_numpy(lfw["images"][2 * p + 1 : 2 * p + 2]).permute(0, 3, 1, 2)
            if rects is not None:
                from .transforms import boxes_to_mask
                m = boxes_to_mask(rects[p : p + 1], 112, 112, "cpu")
                x = x.masked_fill(m, 0)
            axes[i, j].imshow(x[0].permute(1, 2, 0).numpy())
            if i == 0:
                axes[i, j].set_title(cn, fontsize=8)
        if i == 0:
            axes[i, 0].set_title("image 1 (clean)", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("LFW verification pairs: the 2nd image is occluded (test conditions)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ensure_dir("figures") / "occlusion_conditions.png", dpi=200)
    plt.close(fig)
    if verbose:
        print(f"[prepare_data] train {stats['train_images']} images / {n_train_cls} ids; held-out "
              f"{stats['holdout_identities']} ids / {stats['holdout_images']} images -> {stats['val_pairs']} val pairs")
    return stats


# =============================================================================
# loading
# =============================================================================
class MemmapBatches(torch.utils.data.Dataset):
    """Item i = uint8 tensor (B,112,112,3) of the i-th precomputed index batch.
    The memmap is opened lazily inside each DataLoader worker."""

    def __init__(self, path: str, batches: list[np.ndarray]):
        self.path, self.batches, self.arr = path, batches, None

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        if self.arr is None:
            self.arr = np.load(self.path, mmap_mode="r")
        return torch.from_numpy(np.ascontiguousarray(self.arr[self.batches[i]]))


def _loader(cfg, batches):
    path = str(casia_paths(cfg["data"]["cache_dir"])["images"])
    nw = cfg["data"]["num_workers"]
    return torch.utils.data.DataLoader(
        MemmapBatches(path, batches), batch_size=None, shuffle=False, num_workers=nw, pin_memory=True,
        prefetch_factor=4 if nw > 0 else None, persistent_workers=False,
    )


def _epoch_batches(train_idx: np.ndarray, bs: int, g: torch.Generator) -> list[np.ndarray]:
    perm = train_idx[torch.randperm(len(train_idx), generator=g).numpy()]
    n = len(perm) // bs                                   # drop_last
    return [np.sort(perm[s * bs : (s + 1) * bs]) for s in range(n)]   # sorted -> sequential disk reads


# =============================================================================
# training
# =============================================================================
def _param_groups(model: nn.Module, head: nn.Module, wd: float):
    decay, no_decay = [], []
    for p in list(model.parameters()) + list(head.parameters()):
        (decay if p.ndim > 1 else no_decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def _lr_lambda(total: int, warmup: int):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
    return f


def _amp_dtype(name: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(name)


def _train_step(model, head, opt, sched, x8, y, boxes, flip, cfg):
    tcfg = cfg["train"]
    dt = _amp_dtype(tcfg["amp"])
    x = preprocess(x8, boxes, cfg["occlusion"]["fill"], flip)
    if tcfg["channels_last"]:
        x = x.contiguous(memory_format=torch.channels_last)
    with torch.autocast(device_type=x.device.type, dtype=dt, enabled=dt is not None):
        emb = model(x)
    logits, cos = head(emb, y)
    loss = F.cross_entropy(logits, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    return loss.detach(), (cos.detach().argmax(1) == y).sum()


def _build(cfg, n_cls, dev, seed):
    mcfg, tcfg = cfg["model"], cfg["train"]
    set_seed(seed, tcfg["deterministic"])           # identical initial head for clean / aug
    model = build_face_resnet18(mcfg["embedding_dim"], pretrained=True).to(dev)
    head = CosFaceHead(mcfg["embedding_dim"], n_cls, mcfg["scale"], mcfg["margin"]).to(dev)
    if tcfg["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    return model, head


def throughput_test(cfg: dict | None = None, steps: int = 60, mode: str = "aug", deterministic: bool | None = None) -> dict:
    """Measure loader-only and full-training-step throughput (images/s) on random batches."""
    cfg, seed = _cfg_seed(cfg, None)
    if deterministic is not None:
        cfg = {**cfg, "train": {**cfg["train"], "deterministic": deterministic}}
    tcfg, ocfg = cfg["train"], cfg["occlusion"]
    dev = get_device(tcfg["device"])
    _, labels = load_casia(cfg["data"]["cache_dir"])
    split = load_casia_split(seed)
    train_idx, ylab, n_cls = train_indices_and_labels(labels, split["holdout_ids"])
    g = torch.Generator().manual_seed(seed)
    batches = _epoch_batches(train_idx, tcfg["batch_size"], g)[: steps + 10]
    occ_rng = np.random.default_rng(0)

    t0 = time.perf_counter()
    for i, xb in enumerate(_loader(cfg, batches[:steps])):
        pass
    loader_ips = steps * tcfg["batch_size"] / (time.perf_counter() - t0)

    model, head = _build(cfg, n_cls, dev, seed)
    model.train()
    opt = torch.optim.SGD(_param_groups(model, head, tcfg["weight_decay"]), lr=tcfg["lr"], momentum=tcfg["momentum"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(10**6, 10**3))
    times = []
    for i, xb in enumerate(_loader(cfg, batches)):
        x8 = xb.to(dev, non_blocking=True).permute(0, 3, 1, 2)
        y = torch.as_tensor(ylab[batches[i]], device=dev)
        boxes = sample_train_boxes(len(y), ocfg["train_prob"], tuple(ocfg["train_ratio"]), 112, 112, occ_rng) if mode == "aug" else None
        flip = torch.rand(len(y)) < 0.5
        with Timer(dev) as t:
            _train_step(model, head, opt, sched, x8, y, boxes, flip, cfg)
        times.append(t.elapsed)
    step_s = float(np.median(times[10:]))
    res = {
        "loader_images_per_s": loader_ips,
        "train_step_images_per_s_gpu_only": tcfg["batch_size"] / step_s,
        "train_images": int(len(train_idx)),
        "est_epoch_min_gpu_bound": len(train_idx) / (tcfg["batch_size"] / step_s) / 60,
        "est_epoch_min_loader_bound": len(train_idx) / loader_ips / 60,
        "batch_size": tcfg["batch_size"],
        "amp": tcfg["amp"],
        "deterministic": tcfg["deterministic"],
        "num_workers": cfg["data"]["num_workers"],
        "gpu_mem_peak_gb": torch.cuda.max_memory_allocated() / 1024**3 if dev.type == "cuda" else None,
    }
    del model, head, opt
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    print("[throughput] " + ", ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()))
    return res


def _val_data(cfg, seed):
    """Held-out CASIA validation pairs, images preloaded into RAM."""
    images, _ = load_casia(cfg["data"]["cache_dir"])
    split = load_casia_split(seed)
    uniq, inv = np.unique(split["val_pairs"], return_inverse=True)
    imgs = np.ascontiguousarray(images[uniq])
    pairs = inv.reshape(split["val_pairs"].shape)
    conds = pair_conditions(len(pairs), cfg["occlusion"]["val_ratios"], [], dataset_seed(cfg["occlusion"]["val_seed"], f"casia{seed}"))
    return imgs, pairs, split["val_issame"], split["fold"], conds


def _embed_kw(cfg, dev):
    t = cfg["train"]
    return dict(device=dev, batch_size=t["eval_batch_size"], flip=t["eval_flip"], fill=cfg["occlusion"]["fill"],
                amp=t["amp"], channels_last=t["channels_last"])


def _strip(res: dict) -> dict:
    """Drop per-pair arrays from evaluate_pairs output (kept separately in npz)."""
    return {k: {m: v for m, v in r.items() if m != "correct"} for k, r in res.items() if not k.startswith("_")}


def run_dense(
    mode: str,
    cfg: dict | None = None,
    seed: int | None = None,
    epochs: int | None = None,
    tag: str = "",
    resume: bool = True,
    skip_if_exists: bool = False,
    max_train_images: int | None = None,
) -> dict:
    """Train Dense-Clean (mode='clean') or Dense-Aug (mode='aug') on CASIA-WebFace with CosFace.

    Every epoch: verification on held-out CASIA pairs (clean + random squares), best epoch
    (mean accuracy) saved to checkpoints/<name>.pt, full state to checkpoints/<name>_last.pt
    (resume=True continues from it). Then the selected checkpoint is evaluated on
    LFW / AgeDB-30 / CALFW / CPLFW under every occlusion condition.
    """
    assert mode in METHOD_NAME
    cfg, seed = _cfg_seed(cfg, seed)
    cfg = {**cfg, "train": {**cfg["train"]}}
    if epochs is not None:
        cfg["train"]["epochs"] = epochs
    tcfg, ocfg = cfg["train"], cfg["occlusion"]
    name = run_name(mode, seed, tag)
    out_path = resolve(f"results/dense/{name}.json")
    if skip_if_exists and out_path.exists():
        print(f"[run_dense] {name}: result exists -> loading (skip_if_exists=False to retrain)")
        return load_json(out_path)

    logger = get_logger(name, f"results/logs/{name}.log")
    dev = get_device(tcfg["device"])
    env = env_info()
    logger.info(f"run {name} | device {dev} | gpu {env.get('gpu')} | torch {env.get('torch')}")

    _, labels = load_casia(cfg["data"]["cache_dir"])
    split = load_casia_split(seed)
    check_casia_split(split, labels)
    train_idx, ylab, n_cls = train_indices_and_labels(labels, split["holdout_ids"])
    if max_train_images is not None:           # smoke tests only
        train_idx = np.sort(np.random.default_rng(0).choice(train_idx, max_train_images, replace=False))
    vimgs, vpairs, vsame, vfold, vconds = _val_data(cfg, seed)
    ekw = _embed_kw(cfg, dev)
    logger.info(f"train {len(train_idx)} images / {n_cls} identities | val {len(vpairs)} pairs on "
                f"{len(split['holdout_ids'])} held-out identities | conditions {list(vconds)}")

    model, head = _build(cfg, n_cls, dev, seed)
    E, bs = tcfg["epochs"], tcfg["batch_size"]
    steps_per_epoch = len(train_idx) // bs
    opt = torch.optim.SGD(_param_groups(model, head, tcfg["weight_decay"]), lr=tcfg["lr"], momentum=tcfg["momentum"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(E * steps_per_epoch, int(tcfg["warmup_epochs"] * steps_per_epoch)))
    g = torch.Generator().manual_seed(seed)            # batch order + flips
    occ_rng = np.random.default_rng([seed, 7])         # training occlusion boxes
    history, best, start_ep, train_time = [], {"score": -1.0, "epoch": -1}, 1, 0.0
    ckpt_best = resolve(f"checkpoints/{name}.pt")
    ckpt_last = resolve(f"checkpoints/{name}_last.pt")

    if resume and ckpt_last.exists():
        st = torch.load(ckpt_last, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        head.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        g.set_state(st["g_state"])
        occ_rng.bit_generator.state = st["occ_rng_state"]
        history, best, start_ep, train_time = st["history"], st["best"], st["epoch"] + 1, st["train_time"]
        logger.info(f"resumed from {ckpt_last.name}: epoch {st['epoch']} done, best epoch {best['epoch']}")

    t_wall = time.time()
    for ep in range(start_ep, E + 1):
        model.train()
        head.train()
        batches = _epoch_batches(train_idx, bs, g)
        loss_sum = torch.zeros((), device=dev)
        correct = torch.zeros((), device=dev, dtype=torch.long)
        seen, n_occ, t_log = 0, 0, time.perf_counter()
        with Timer(dev) as t:
            for s, xb in enumerate(_loader(cfg, batches)):
                bi = batches[s]
                x8 = xb.to(dev, non_blocking=True).permute(0, 3, 1, 2)
                y = torch.as_tensor(ylab[bi], device=dev)
                boxes = None
                if mode == "aug":
                    boxes = sample_train_boxes(len(bi), ocfg["train_prob"], tuple(ocfg["train_ratio"]), 112, 112, occ_rng)
                    n_occ += int((boxes[:, 2] > 0).sum())
                flip = (torch.rand(len(bi), generator=g) < 0.5) if tcfg["hflip"] else None
                loss, corr = _train_step(model, head, opt, sched, x8, y, boxes, flip, cfg)
                loss_sum += loss * len(bi)
                correct += corr
                seen += len(bi)
                if (s + 1) % tcfg["log_every"] == 0:
                    ips = tcfg["log_every"] * bs / (time.perf_counter() - t_log)
                    t_log = time.perf_counter()
                    logger.info(f"[{mode}] ep {ep:02d} step {s + 1}/{steps_per_epoch} loss {loss_sum.item() / seen:.4f} "
                                f"acc {correct.item() / seen:.4f} lr {opt.param_groups[0]['lr']:.4f} | {ips:.0f} img/s")
                    if not math.isfinite(loss.item()):
                        raise FloatingPointError("non-finite loss")
        train_time += t.elapsed

        with Timer(dev) as tv:
            val = evaluate_pairs(model, vimgs, vpairs, vsame, vconds, vfold, **ekw)
        accs = {k: v["acc"] for k, v in val.items() if not k.startswith("_")}
        score = float(np.mean(list(accs.values())))
        rec = {
            "epoch": ep,
            "train_loss": loss_sum.item() / seen,
            "train_acc": correct.item() / seen,
            "train_occluded_frac": n_occ / seen,
            "lr_end": opt.param_groups[0]["lr"],
            "val_acc": accs,
            "val_auc": {k: v["auc"] for k, v in val.items() if not k.startswith("_")},
            "val_select_score": score,
            "epoch_train_time_s": t.elapsed,
            "val_time_s": tv.elapsed,
        }
        history.append(rec)
        improved = score > best["score"]
        if improved:
            best = {"score": score, "epoch": ep}
            torch.save({"model": model.state_dict(), "head": head.state_dict(),
                        "meta": {"name": name, "mode": mode, "seed": seed, "epoch": ep, "val_acc": accs,
                                 "n_classes": n_cls, "config": cfg}}, ckpt_best)
        torch.save({"model": model.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "g_state": g.get_state(), "occ_rng_state": occ_rng.bit_generator.state,
                    "history": history, "best": best, "epoch": ep, "train_time": train_time}, ckpt_last)
        logger.info(f"[{mode}] ep {ep:02d}/{E} loss {rec['train_loss']:.4f} train_acc {rec['train_acc']:.4f} | val "
                    + " ".join(f"{k} {v:.4f}" for k, v in accs.items())
                    + f" | sel {score:.4f}{' *' if improved else ''} | train {t.elapsed / 60:.1f} min, val {tv.elapsed:.0f}s")

    logger.info(f"training done: best epoch {best['epoch']} (val score {best['score']:.4f}), "
                f"pure train time {train_time / 60:.1f} min")
    del opt, vimgs
    result = evaluate_checkpoint(ckpt_best, cfg, seed, logger=logger)
    result.update({
        "name": name, "method": METHOD_NAME[mode], "mode": mode, "seed": seed, "sparsity": 0.0,
        "best_epoch": best["epoch"], "best_val_score": best["score"], "select_metric": tcfg["select_metric"],
        "history": history, "n_train_images": int(len(train_idx)), "n_train_identities": int(n_cls),
        "timing": {"train_time_s": train_time, "wall_time_this_session_s": time.time() - t_wall, "device": str(dev),
                   "note": "train_time_s sums per-epoch training loops (accelerator-synchronised), across resumes"},
        "checkpoint": str(ckpt_best.relative_to(resolve("."))), "config": cfg, "env": env,
    })
    save_json(result, out_path)
    logger.info(f"saved {out_path}")
    del model, head
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return load_json(out_path)


# =============================================================================
# evaluation
# =============================================================================
def load_checkpoint(path, device=None, *, weights_only: bool = False) -> tuple[nn.Module, nn.Module, dict]:
    device = device or get_device("auto")
    st = torch.load(resolve(path), map_location=device, weights_only=weights_only)
    meta = st["meta"]
    mcfg = meta["config"]["model"]
    model = build_face_resnet18(mcfg["embedding_dim"], pretrained=False)
    model.load_state_dict(st["model"])
    head = CosFaceHead(mcfg["embedding_dim"], meta["n_classes"], mcfg["scale"], mcfg["margin"])
    head.load_state_dict(st["head"])
    model, head = model.to(device), head.to(device)
    if meta["config"]["train"]["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    return model, head, meta


def train_diagnostics(model, head, cfg, seed, n_batches: int = 8, batch_size: int = 32) -> dict:
    """Clean vs occluded CosFace loss / accuracy / conv-weight gradient norm on fixed training
    images (BN in eval mode) - the signal Taylor importance scores are built from."""
    ocfg = cfg["occlusion"]
    dev = next(model.parameters()).device
    images, labels = load_casia(cfg["data"]["cache_dir"])
    split = load_casia_split(seed)
    train_idx, ylab, _ = train_indices_and_labels(labels, split["holdout_ids"])
    rng = np.random.default_rng(ocfg["diag_seed"])
    sub = np.sort(rng.choice(train_idx, n_batches * batch_size, replace=False))
    boxes_all = sample_train_boxes(len(sub), 1.0, tuple(ocfg["train_ratio"]), 112, 112, rng)
    model.eval()
    head.eval()
    convs = [m.weight for m in model.modules() if isinstance(m, nn.Conv2d)]
    res = {}
    for cond in ("clean", "occluded"):
        losses, accs, norms = [], [], []
        for b in range(n_batches):
            sl = slice(b * batch_size, (b + 1) * batch_size)
            x8 = torch.from_numpy(np.ascontiguousarray(images[sub[sl]])).to(dev).permute(0, 3, 1, 2)
            y = torch.as_tensor(ylab[sub[sl]], device=dev)
            x = preprocess(x8, None if cond == "clean" else boxes_all[sl], ocfg["fill"])
            model.zero_grad(set_to_none=True)
            logits, cos = head(model(x), y)            # fp32, no autocast
            loss = F.cross_entropy(logits, y)
            loss.backward()
            norms.append(float(sum((p.grad.detach() ** 2).sum() for p in convs)) ** 0.5)
            losses.append(loss.item())
            accs.append(float((cos.argmax(1) == y).float().mean()))
        res[f"train_{cond}_loss"] = float(np.mean(losses))
        res[f"train_{cond}_acc"] = float(np.mean(accs))
        res[f"grad_norm_{cond}_mean"] = float(np.mean(norms))
        res[f"grad_norm_{cond}_std"] = float(np.std(norms))
    model.zero_grad(set_to_none=True)
    res["grad_norm_ratio_occ_over_clean"] = res["grad_norm_occluded_mean"] / res["grad_norm_clean_mean"]
    res.update(n_batches=n_batches, batch_size=batch_size)
    return res


def evaluate_checkpoint(ckpt_path, cfg: dict | None = None, seed: int | None = None, logger=None,
                             save_sims_as: str | None = None) -> dict:
    """Evaluate a checkpoint on held-out CASIA pairs and every test set x occlusion condition.
    Per-pair cosine similarities are saved to results/dense/<name>_pair_sims.npz."""
    cfg, seed = _cfg_seed(cfg, seed)
    dcfg, ocfg = cfg["data"], cfg["occlusion"]
    dev = get_device(cfg["train"]["device"])
    model, head, meta = load_checkpoint(ckpt_path, dev)
    ekw = _embed_kw(cfg, dev)
    log = logger.info if logger is not None else print
    out, sims = {"test": {}}, {}
    vimgs, vpairs, vsame, vfold, vconds = _val_data(cfg, seed)
    val = evaluate_pairs(model, vimgs, vpairs, vsame, vconds, vfold, **ekw)
    out["val"] = _strip(val)
    log("casia-val: " + " | ".join(f"{k} {v['acc']:.4f}" for k, v in out["val"].items()))
    with Timer(dev) as te:
        for name in dcfg["test_sets"]:
            d = load_pairs(dcfg["cache_dir"], name)
            pairs = np.arange(len(d["images"])).reshape(-1, 2)
            conds = pair_conditions(len(pairs), ocfg["test_ratios"], ocfg["test_regions"], dataset_seed(ocfg["test_seed"], name))
            r = evaluate_pairs(model, d["images"], pairs, d["issame"], conds, None, **ekw)
            for c, s in r["_sims"].items():
                sims[f"{name}/{c}"] = s
            out["test"][name] = _strip(r)
            log(f"{name}: " + " | ".join(f"{k} {v['acc'] * 100:.2f}+-{v['acc_std'] * 100:.2f}" for k, v in out["test"][name].items()))
    out["test_eval_time_s"] = te.elapsed
    out["train_diagnostics"] = train_diagnostics(model, head, cfg, seed)
    log("diagnostics: " + ", ".join(f"{k}={v:.4g}" for k, v in out["train_diagnostics"].items()))
    nz = sum(int((p != 0).sum()) for p in model.parameters())
    out["params"] = {"embedding_net_total": sum(p.numel() for p in model.parameters()), "embedding_net_nonzero": nz,
                     "cosface_head": head.weight.numel()}
    stem = save_sims_as or meta["name"]
    ensure_dir("results/dense")
    np.savez_compressed(resolve(f"results/dense/{stem}_pair_sims.npz"), **sims,
                        **{f"{n}/issame": load_pairs(dcfg["cache_dir"], n)["issame"] for n in dcfg["test_sets"]})
    return out


# =============================================================================
# summary
# =============================================================================
COND_LABEL = {"clean": "Clean", "square10": "Sq 10%", "square30": "Sq 30%", "square50": "Sq 50%",
              "eyes": "Eyes", "mouth_nose": "Mouth+nose", "upper_half": "Upper half", "lower_half": "Lower half"}


def paired_bootstrap_pairs(sims_a: np.ndarray, sims_b: np.ndarray, issame: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """95% CI of acc(B) - acc(A) over resampled pairs, each model using its own 10-fold thresholds."""
    from .faceverif import verification_metrics

    ca = verification_metrics(sims_a, issame)["correct"].astype(float)
    cb = verification_metrics(sims_b, issame)["correct"].astype(float)
    d = cb - ca
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {"diff": float(d.mean()), "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5))}


def summarize_dense(seed: int = 42, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    runs = {}
    for mode in METHOD_NAME:
        p = resolve(f"results/dense/{run_name(mode, seed)}.json")
        if p.exists():
            r = load_json(p)
            runs[r["method"]] = r
        elif verbose:
            print(f"[summarize_dense] missing {p.name}")
    if not runs:
        raise FileNotFoundError("no dense results - run run_dense('clean') / ('aug') first")
    rows = []
    for m, r in runs.items():
        lfw = r["test"]["lfw"]
        row = {"Method": m, "Best epoch": r["best_epoch"]}
        row.update({f"LFW {COND_LABEL[c]}": f"{100 * v['acc']:.2f} ± {100 * v['acc_std']:.2f}" for c, v in lfw.items()})
        for n in ("agedb_30", "calfw", "cplfw"):
            if n in r["test"]:
                row[f"{n.upper()} Clean"] = f"{100 * r['test'][n]['clean']['acc']:.2f}"
        row["CASIA-val mean"] = f"{100 * r['best_val_score']:.2f}"
        row["Train time (h)"] = round(r["timing"]["train_time_s"] / 3600, 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    out_dir = ensure_dir("results")
    df.to_csv(out_dir / f"baseline_table_seed{seed}.csv", index=False)
    md = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    md += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    (out_dir / f"baseline_table_seed{seed}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    figs = {}
    fig_dir = ensure_dir("figures")
    conds = list(next(iter(runs.values()))["test"]["lfw"].keys())
    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = 0.8 / len(runs)
    for i, (m, r) in enumerate(runs.items()):
        ys = [100 * r["test"]["lfw"][c]["acc"] for c in conds]
        es = [100 * r["test"]["lfw"][c]["acc_std"] for c in conds]
        xs = np.arange(len(conds)) + (i - (len(runs) - 1) / 2) * w
        ax.bar(xs, ys, w, yerr=es, capsize=2, color=COLORS[m], label=m)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax.axvline(3.5, color="gray", lw=0.8, ls="--")
    ax.set_xticks(range(len(conds)), [COND_LABEL[c] for c in conds])
    ax.set_ylabel("LFW verification accuracy (%)")
    ax.set_ylim(50, 100.5)
    ax.set_title(f"CASIA-WebFace -> LFW, 2nd image of each pair occluded (10-fold mean ± std, seed {seed})")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(fig_dir / f"dense_lfw_seed{seed}.png", dpi=200)
    fig.savefig(fig_dir / f"dense_lfw_seed{seed}.pdf")
    plt.close(fig)
    figs["lfw_conditions"] = fig

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for m, r in runs.items():
        h = r["history"]
        ep = [x["epoch"] for x in h]
        axes[0].plot(ep, [x["train_loss"] for x in h], color=COLORS[m], label=m)
        axes[1].plot(ep, [100 * x["val_acc"]["clean"] for x in h], color=COLORS[m], label=f"{m} clean")
        axes[1].plot(ep, [100 * x["val_acc"]["square30"] for x in h], color=COLORS[m], ls="--", label=f"{m} sq 30%")
        axes[1].axvline(r["best_epoch"], color=COLORS[m], lw=0.8, ls=":")
    axes[0].set_title("training CosFace loss")
    axes[1].set_title("held-out CASIA verification accuracy (%)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"dense_training_curves_seed{seed}.png", dpi=200)
    plt.close(fig)
    figs["training_curves"] = fig

    if len(runs) == 2:
        za = np.load(resolve(f"results/dense/{run_name('clean', seed)}_pair_sims.npz"))
        zb = np.load(resolve(f"results/dense/{run_name('aug', seed)}_pair_sims.npz"))
        boot = {c: paired_bootstrap_pairs(za[f"lfw/{c}"], zb[f"lfw/{c}"], za["lfw/issame"]) for c in conds}
        save_json(boot, f"results/dense/bootstrap_lfw_seed{seed}.json")
        figs["bootstrap"] = boot
    return df, figs


# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Occlusion-aware pruning experiments (CASIA-WebFace -> LFW)")
    ap.add_argument("cmd", choices=["prepare", "throughput", "train", "summarize"])
    ap.add_argument("--mode", choices=list(METHOD_NAME), default="aug")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--max-train-images", type=int, default=None)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--nondeterministic", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.nondeterministic:
        cfg["train"]["deterministic"] = False
    if a.cmd == "prepare":
        prepare_data(cfg, a.seed)
    elif a.cmd == "throughput":
        throughput_test(cfg, a.steps, a.mode)
    elif a.cmd == "train":
        run_dense(a.mode, cfg, a.seed, a.epochs, a.tag, resume=not a.no_resume, max_train_images=a.max_train_images)
    else:
        print(summarize_dense(a.seed or cfg["seed"])[0].T)


if __name__ == "__main__":
    main()
