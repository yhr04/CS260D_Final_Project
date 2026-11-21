"""Dense fine-tuning loop (Dense-Clean / Dense-Aug).

The same loop will be reused for post-pruning fine-tuning (with masks applied
after each optimizer step), so every method shares one training procedure.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .evaluate import evaluate_levels
from .occlusion import sample_train_boxes
from .transforms import preprocess
from .utils import Timer


def make_optimizer(model: nn.Module, tcfg: dict) -> torch.optim.Optimizer:
    """AdamW with a higher LR for the new head; no weight decay on BN / biases."""
    head_ids = {id(p) for p in model.fc.parameters()}
    groups = {"bb_decay": [], "bb_no_decay": [], "head_decay": [], "head_no_decay": []}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        part = "head" if id(p) in head_ids else "bb"
        decay = "decay" if p.ndim > 1 else "no_decay"
        groups[f"{part}_{decay}"].append(p)
    wd = tcfg["weight_decay"]
    param_groups = [
        {"params": groups["bb_decay"], "lr": tcfg["lr_backbone"], "weight_decay": wd},
        {"params": groups["bb_no_decay"], "lr": tcfg["lr_backbone"], "weight_decay": 0.0},
        {"params": groups["head_decay"], "lr": tcfg["lr_head"], "weight_decay": wd},
        {"params": groups["head_no_decay"], "lr": tcfg["lr_head"], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW([g for g in param_groups if g["params"]])


def cosine_lambda(total_steps: int) -> Callable[[int], float]:
    return lambda step: 0.5 * (1.0 + math.cos(math.pi * min(step, total_steps) / max(1, total_steps)))


def train_model(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    split: dict,
    val_boxes: dict,
    cfg: dict,
    mode: str,
    seed: int,
    logger,
    on_new_best: Callable[[nn.Module, int, dict], None] | None = None,
    after_step: Callable[[nn.Module], None] | None = None,
    epochs: int | None = None,
) -> dict:
    """Train `model`; mode='clean' (no occlusion) or 'aug' (random occlusion w.p. train_prob).

    Model selection: epoch maximising cfg.train.select_metric on the validation
    set (never the test set). `on_new_best` is called to checkpoint.
    `after_step` (e.g. re-applying pruning masks) is called after every optimizer step.
    """
    assert mode in ("clean", "aug")
    tcfg, ocfg = cfg["train"], cfg["occlusion"]
    img_size = cfg["data"]["img_size"]
    device = images.device
    H, W = images.shape[2], images.shape[3]
    epochs = epochs or tcfg["epochs"]
    bs = tcfg["batch_size"]
    train_idx = split["train"]
    steps_per_epoch = len(train_idx) // bs  # drop_last (avoids BN on a batch of 1)

    opt = make_optimizer(model, tcfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lambda(epochs * steps_per_epoch))
    g = torch.Generator().manual_seed(seed)           # batch order + flips
    occ_rng = np.random.default_rng([seed, 7])        # training occlusion masks

    history, best = [], {"score": -1.0, "epoch": -1}
    total_train_time = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), generator=g).numpy()]
        loss_sum, correct, seen, n_occ = 0.0, 0, 0, 0
        with Timer(device) as t:
            for s in range(steps_per_epoch):
                bi = perm[s * bs : (s + 1) * bs]
                bt = torch.as_tensor(bi, device=device)
                boxes = None
                if mode == "aug":
                    boxes = sample_train_boxes(len(bi), ocfg["train_prob"], tuple(ocfg["train_ratio"]), H, W, occ_rng)
                    n_occ += int((boxes[:, 2] > 0).sum())
                flip = (torch.rand(len(bi), generator=g) < 0.5) if tcfg.get("hflip", True) else None
                x = preprocess(images[bt], boxes, ocfg["fill"], flip, img_size)
                y = labels[bt]
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                if after_step is not None:
                    after_step(model)
                loss_sum += loss.item() * len(bi)
                correct += int((logits.argmax(1) == y).sum())
                seen += len(bi)
        total_train_time += t.elapsed

        with Timer(device) as tv:
            val = evaluate_levels(model, images, labels, split["val"], val_boxes, img_size, ocfg["fill"], tcfg["eval_batch_size"])
        val_accs = {k: v["acc"] for k, v in val["levels"].items()}
        score = val["mean_acc"] if tcfg["select_metric"] == "val_mean_acc" else val_accs[0.0]
        rec = {
            "epoch": ep,
            "train_loss": loss_sum / seen,
            "train_acc": correct / seen,
            "train_occluded_frac": n_occ / seen,
            "lr_backbone": opt.param_groups[0]["lr"],
            "val_acc": val_accs,
            "val_bal_acc": {k: v["bal_acc"] for k, v in val["levels"].items()},
            "val_loss": {k: v["loss"] for k, v in val["levels"].items()},
            "val_select_score": score,
            "epoch_train_time_s": t.elapsed,
            "val_time_s": tv.elapsed,
        }
        history.append(rec)
        improved = score > best["score"]
        if improved:
            best = {"score": score, "epoch": ep}
            if on_new_best is not None:
                on_new_best(model, ep, rec)
        logger.info(
            f"[{mode}] ep {ep:02d}/{epochs} loss {rec['train_loss']:.4f} train_acc {rec['train_acc']:.3f} | "
            + " ".join(f"val@{int(k*100)}% {v:.3f}" for k, v in val_accs.items())
            + f" | sel {score:.4f}{' *' if improved else ''} | {t.elapsed:.1f}s"
        )
    return {"history": history, "best_epoch": best["epoch"], "best_val_score": best["score"], "train_time_s": total_train_time}
