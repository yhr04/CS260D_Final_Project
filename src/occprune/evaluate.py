"""Evaluation on fixed occlusion masks + training-set diagnostics."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

from .occlusion import sample_train_boxes
from .transforms import preprocess


@torch.no_grad()
def predict(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    idx: np.ndarray,
    boxes: np.ndarray | None,
    img_size: int,
    fill: float = 0.0,
    batch_size: int = 128,
) -> tuple[np.ndarray, float]:
    """Predictions for images[idx] with optional per-image boxes (len(idx),3).
    Returns (predicted labels, mean cross-entropy loss)."""
    model.eval()
    preds, loss_sum = [], 0.0
    for s in range(0, len(idx), batch_size):
        bi = idx[s : s + batch_size]
        bt = torch.as_tensor(bi, device=images.device)
        x = preprocess(images[bt], None if boxes is None else boxes[s : s + batch_size], fill, None, img_size)
        logits = model(x)
        y = labels[bt]
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds), loss_sum / len(idx)


def evaluate_levels(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    idx: np.ndarray,
    eval_boxes: dict,
    img_size: int,
    fill: float = 0.0,
    batch_size: int = 128,
    keep_preds: bool = False,
) -> dict:
    """Accuracy / balanced accuracy / loss at every occlusion level.

    eval_boxes: {ratio: (R, len(idx), 3)}. Metrics are averaged over the R
    mask repeats; std over repeats is reported too (this is mask variability,
    NOT sampling uncertainty over original images).
    """
    y_true = labels[torch.as_tensor(idx, device=labels.device)].cpu().numpy()
    out = {"levels": {}, "preds": {} if keep_preds else None}
    for ratio, reps in sorted(eval_boxes.items()):
        accs, baccs, losses, plist = [], [], [], []
        for r in range(reps.shape[0]):
            boxes = None if ratio <= 0 else reps[r]
            p, loss = predict(model, images, labels, idx, boxes, img_size, fill, batch_size)
            accs.append(float((p == y_true).mean()))
            baccs.append(float(balanced_accuracy_score(y_true, p)))
            losses.append(loss)
            plist.append(p)
        out["levels"][float(ratio)] = {
            "acc": float(np.mean(accs)),
            "acc_std_over_masks": float(np.std(accs)),
            "bal_acc": float(np.mean(baccs)),
            "bal_acc_std_over_masks": float(np.std(baccs)),
            "loss": float(np.mean(losses)),
            "acc_per_mask": accs,
            "n_images": int(len(idx)),
            "n_masks": int(reps.shape[0]),
        }
        if keep_preds:
            out["preds"][float(ratio)] = np.stack(plist).astype(np.int16)
    accs = [v["acc"] for v in out["levels"].values()]
    out["mean_acc"] = float(np.mean(accs))
    clean = out["levels"].get(0.0, {}).get("acc")
    if clean is not None:
        out["robustness_drop"] = {k: clean - v["acc"] for k, v in out["levels"].items() if k > 0}
    out["y_true"] = y_true if keep_preds else None
    return out


def train_set_diagnostics(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    train_idx: np.ndarray,
    cfg: dict,
    n_grad_batches: int = 8,
    grad_batch_size: int = 16,
) -> dict:
    """Diagnostics that matter for gradient-based (Taylor) importance later on.

    A model fine-tuned to near-zero training loss has tiny, noisy clean
    gradients, which makes |w * dL/dw| a weak signal. We measure training-set
    loss/accuracy and the gradient norm (BN in eval mode) for clean vs.
    occluded training images under a fixed seed.
    """
    ocfg = cfg["occlusion"]
    img_size = cfg["data"]["img_size"]
    H, W = images.shape[2], images.shape[3]
    rng = np.random.default_rng(ocfg["diag_seed"])
    occ_boxes = sample_train_boxes(len(train_idx), 1.0, tuple(ocfg["train_ratio"]), H, W, rng)
    res = {}
    for name, boxes in (("clean", None), ("occluded", occ_boxes)):
        p, loss = predict(model, images, labels, train_idx, boxes, img_size, ocfg["fill"])
        y = labels[torch.as_tensor(train_idx, device=labels.device)].cpu().numpy()
        res[f"train_{name}_acc"] = float((p == y).mean())
        res[f"train_{name}_loss"] = float(loss)

    # gradient norm of the mean mini-batch loss w.r.t. conv weights
    model.eval()
    conv_params = [m.weight for m in model.modules() if isinstance(m, nn.Conv2d)]
    sub = np.random.default_rng(ocfg["diag_seed"] + 1).permutation(train_idx)[: n_grad_batches * grad_batch_size]
    sub_pos = {int(v): i for i, v in enumerate(train_idx)}
    for name in ("clean", "occluded"):
        norms = []
        for b in range(n_grad_batches):
            bi = sub[b * grad_batch_size : (b + 1) * grad_batch_size]
            bt = torch.as_tensor(bi, device=images.device)
            boxes = None if name == "clean" else occ_boxes[[sub_pos[int(v)] for v in bi]]
            x = preprocess(images[bt], boxes, ocfg["fill"], None, img_size)
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), labels[bt]).backward()
            g2 = sum(float((p.grad.detach() ** 2).sum()) for p in conv_params)
            norms.append(g2**0.5)
        res[f"grad_norm_{name}_mean"] = float(np.mean(norms))
        res[f"grad_norm_{name}_std"] = float(np.std(norms))
    model.zero_grad(set_to_none=True)
    res["grad_batch_size"] = grad_batch_size
    res["n_grad_batches"] = n_grad_batches
    return res
