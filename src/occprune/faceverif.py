"""Model pieces and the face-verification protocol.

* Model: ImageNet ResNet-18 trunk, global pool, then ``model.fc`` = FC(512->512, no bias) + BN1d
  gives the 512-d embedding. The 19 prunable convs keep torchvision's layer names.
* Loss: CosFace (large-margin cosine), head kept separate from the embedding network.
* Verification: cosine similarity of L2-normalised embeddings (flip-augmented), accuracy with
  the standard 10-fold protocol (threshold chosen on 9 folds, accuracy on the held-out fold).
* Occluded pairs: the SECOND image of every pair is occluded (occluded probe vs clean gallery).
  Random squares use fixed, saved boxes; region occlusions use the fixed 112x112 alignment
  template (eyes / nose / mouth sit at the same pixels in every aligned face).
"""
from __future__ import annotations

import math
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .occlusion import patch_side
from .transforms import preprocess
from .utils import resolve

# ----------------------------------------------------------------------------- model
def build_face_resnet18(embedding_dim: int = 512, pretrained: bool = True) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, embedding_dim, bias=False), nn.BatchNorm1d(embedding_dim))
    return model


class CosFaceHead(nn.Module):
    """logits = s * (cos(theta_y) - m) for the target class, s * cos(theta_j) otherwise.
    Computed in fp32 regardless of autocast."""

    def __init__(self, in_features: int, num_classes: int, scale: float = 64.0, margin: float = 0.35):
        super().__init__()
        self.scale, self.margin = float(scale), float(margin)
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.normal_(self.weight, std=0.01)

    def cosine(self, emb: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=emb.device.type, enabled=False):
            return F.linear(F.normalize(emb.float(), dim=1), F.normalize(self.weight.float(), dim=1))

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (margin logits for the loss, plain cosine for accuracy)."""
        cos = self.cosine(emb)
        onehot = F.one_hot(labels, cos.shape[1]).to(cos.dtype)
        return self.scale * (cos - self.margin * onehot), cos


# ----------------------------------------------------------------------------- occlusions
# [y0, x0, h, w] in the 112x112 ArcFace alignment template
# (landmarks: eyes ~(38,52) & (74,52), nose tip ~(56,72), mouth corners ~(42,92) & (71,92)).
REGION_RECTS = {
    "eyes": (36, 18, 30, 76),         # sunglasses-like band over both eyes
    "mouth_nose": (62, 14, 50, 84),   # face-mask-like: nose to chin
    "upper_half": (0, 0, 56, 112),
    "lower_half": (56, 0, 56, 112),
}
REGIONS = tuple(REGION_RECTS)          # held-out region conditions, in reporting order


def region_rects(name: str, n: int) -> np.ndarray:
    return np.tile(np.array(REGION_RECTS[name], dtype=np.int32), (n, 1))


def random_square_rects(n: int, ratio: float, H: int, W: int, seed: int) -> np.ndarray:
    """Fixed random squares of area `ratio`, as (n,4) [y0,x0,side,side]."""
    side = patch_side(ratio, H, W)
    rng = np.random.default_rng([seed, int(round(ratio * 1000))])
    y0 = np.floor(rng.random(n) * (H - side + 1)).astype(np.int32)
    x0 = np.floor(rng.random(n) * (W - side + 1)).astype(np.int32)
    return np.stack([y0, x0, np.full(n, side), np.full(n, side)], 1).astype(np.int32)


def pair_conditions(n_pairs: int, ratios, regions, seed: int, H: int = 112, W: int = 112) -> dict:
    """{condition name: (n_pairs,4) rects or None} for the second image of every pair."""
    conds = {"clean": None}
    for r in ratios:
        conds[f"square{int(round(r * 100))}"] = random_square_rects(n_pairs, r, H, W, seed)
    for g in regions:
        conds[g] = region_rects(g, n_pairs)
    return conds


def dataset_seed(base: int, name: str) -> int:
    return int(base) + zlib.crc32(name.encode()) % 100000


def save_conditions(conds: dict, name: str):
    p = resolve(f"results/masks/{name}_pair_occlusions.npz")
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **{k: v for k, v in conds.items() if v is not None})
    return p


# ----------------------------------------------------------------------------- embeddings
def embed_kwargs(cfg: dict, device) -> dict:
    """Keyword arguments of `embed` / `evaluate_pairs` taken from a config's train section."""
    t = cfg["train"]
    return dict(device=device, batch_size=t["eval_batch_size"], flip=t["eval_flip"], fill=cfg["occlusion"]["fill"],
                amp=t["amp"], channels_last=t["channels_last"])


@torch.no_grad()
def embed(
    model: nn.Module,
    images: np.ndarray,
    idx: np.ndarray | None = None,
    rects: np.ndarray | None = None,
    device=None,
    batch_size: int = 512,
    flip: bool = True,
    fill: float = 0.0,
    amp: str = "bf16",
    channels_last: bool = True,
) -> np.ndarray:
    """L2-normalised embeddings (n,D) float32 of images[idx] (uint8 NHWC), with optional
    per-image occlusion rects (n,4). flip=True adds the embedding of the mirrored image."""
    model.eval()
    device = device or next(model.parameters()).device
    idx = np.arange(len(images)) if idx is None else np.asarray(idx)
    out = []
    dtype = torch.bfloat16 if amp == "bf16" else (torch.float16 if amp == "fp16" else None)
    for s in range(0, len(idx), batch_size):
        bi = idx[s : s + batch_size]
        x8 = torch.from_numpy(np.ascontiguousarray(images[bi])).to(device).permute(0, 3, 1, 2)
        r = None if rects is None else rects[s : s + batch_size]
        xs = [preprocess(x8, r, fill)]
        if flip:
            xs.append(preprocess(x8, r, fill, torch.ones(len(bi), dtype=torch.bool)))
        e = 0
        for x in xs:
            layout = torch.channels_last if channels_last else torch.contiguous_format
            x = x.contiguous(memory_format=layout)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                e = e + model(x).float()
        out.append(F.normalize(e, dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# ----------------------------------------------------------------------------- metrics
def best_threshold(sims: np.ndarray, issame: np.ndarray) -> tuple[float, float]:
    """Threshold maximising accuracy of `sims > t` (exact search over all cut points)."""
    o = np.argsort(sims, kind="stable")
    s, y = sims[o], issame[o].astype(bool)
    n = len(s)
    neg_below = np.concatenate([[0], np.cumsum(~y)])            # negatives among the k smallest
    pos_above = y.sum() - np.concatenate([[0], np.cumsum(y)])   # positives among the rest
    acc = (neg_below + pos_above) / n
    valid = np.ones(n + 1, dtype=bool)
    valid[1:n] = s[1:] > s[:-1]                                  # only cut between distinct values
    acc = np.where(valid, acc, -1)
    k = int(np.argmax(acc))
    lo = s[k - 1] if k > 0 else s[0] - 1e-6
    hi = s[k] if k < n else s[-1] + 1e-6
    return float((lo + hi) / 2), float(acc[k])


def verification_metrics(sims: np.ndarray, issame: np.ndarray, folds: np.ndarray | None = None, n_folds: int = 10) -> dict:
    """10-fold verification accuracy (+ std), TAR@FAR=1e-3 and AUC on all pairs."""
    from sklearn.metrics import roc_auc_score

    sims, issame = np.asarray(sims, dtype=np.float64), np.asarray(issame, dtype=bool)
    if folds is None:
        folds = np.arange(len(sims)) * n_folds // len(sims)   # consecutive folds (LFW order)
    accs, thrs, correct = [], [], np.zeros(len(sims), dtype=bool)
    for k in np.unique(folds):
        te = folds == k
        t, _ = best_threshold(sims[~te], issame[~te])
        pred = sims[te] > t
        correct[te] = pred == issame[te]
        accs.append(float(correct[te].mean()))
        thrs.append(t)
    neg = np.sort(sims[~issame])[::-1]
    far = 1e-3
    k = int(math.floor(far * len(neg)))
    thr_far = neg[k] if k < len(neg) else neg[-1]
    return {
        "acc": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "acc_per_fold": accs,
        "thresholds": thrs,
        "tar_at_far1e-3": float((sims[issame] > thr_far).mean()),
        "auc": float(roc_auc_score(issame, sims)),
        "pos_sim_mean": float(sims[issame].mean()),
        "neg_sim_mean": float(sims[~issame].mean()),
        "n_pairs": int(len(sims)),
        "correct": correct,   # per-pair correctness at the fold threshold (for paired tests)
    }


def evaluate_pairs(
    model: nn.Module,
    images: np.ndarray,
    pairs: np.ndarray,
    issame: np.ndarray,
    conditions: dict,
    folds: np.ndarray | None = None,
    **embed_kw,
) -> dict:
    """images: uint8 NHWC array (or memmap); pairs: (P,2) indices into images.
    The first image is always clean; the second gets each condition's rects.
    Returns {condition: metrics} plus "_sims": {condition: (P,) cosine similarities}."""
    e1 = embed(model, images, pairs[:, 0], None, **embed_kw)
    out, sims_all = {}, {}
    for name, rects in conditions.items():
        e2 = embed(model, images, pairs[:, 1], rects, **embed_kw)
        sims = (e1 * e2).sum(1)
        out[name] = verification_metrics(sims, issame, folds)
        sims_all[name] = sims.astype(np.float32)
    out["_sims"] = sims_all
    return out


def drop_per_pair(res: dict) -> dict:
    """evaluate_pairs output without per-pair arrays ("_sims", "correct"), ready for JSON."""
    return {k: {m: v for m, v in r.items() if m != "correct"} for k, r in res.items() if not k.startswith("_")}
