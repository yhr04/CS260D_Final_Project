"""Training primitives shared by dense training (pipeline.run_dense) and post-pruning
fine-tuning (prune_run): batching from the CASIA memmap, optimizer groups, learning-rate
schedule, one CosFace training step and model construction."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import casia_paths
from .faceverif import CosFaceHead, build_face_resnet18
from .transforms import preprocess
from .utils import set_seed


# ----------------------------------------------------------------------------- data
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


def make_loader(cfg: dict, batches: list[np.ndarray], pin_memory: bool | None = None) -> torch.utils.data.DataLoader:
    """DataLoader yielding the given index batches from the CASIA cache, in order.
    pin_memory defaults to "a CUDA device is available"; pass False when training on the CPU."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    path = str(casia_paths(cfg["data"]["cache_dir"])["images"])
    nw = cfg["data"]["num_workers"]
    return torch.utils.data.DataLoader(
        MemmapBatches(path, batches), batch_size=None, shuffle=False, num_workers=nw,
        pin_memory=pin_memory, prefetch_factor=4 if nw > 0 else None, persistent_workers=False,
    )


def epoch_batches(train_idx: np.ndarray, bs: int, g: torch.Generator) -> list[np.ndarray]:
    """One epoch of shuffled batches (drop_last); indices sorted within a batch for sequential reads."""
    perm = train_idx[torch.randperm(len(train_idx), generator=g).numpy()]
    n = len(perm) // bs
    return [np.sort(perm[s * bs : (s + 1) * bs]) for s in range(n)]


# ----------------------------------------------------------------------------- optimisation
def param_groups(model: nn.Module, head: nn.Module, wd: float) -> list[dict]:
    """Weight decay on weight matrices / kernels only, not on BatchNorm parameters or biases."""
    decay, no_decay = [], []
    for p in list(model.parameters()) + list(head.parameters()):
        (decay if p.ndim > 1 else no_decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def lr_lambda(total: int, warmup: int):
    """Multiplier for LambdaLR: linear warm-up over `warmup` steps, then per-step cosine to 0."""
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
    return f


def amp_dtype(name: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(name)


def train_step(model, head, opt, sched, x8, y, boxes, flip, cfg):
    """One CosFace step on a uint8 batch (occlusion boxes and flips applied here).
    Returns (detached loss, number of correct argmax predictions)."""
    tcfg = cfg["train"]
    dt = amp_dtype(tcfg["amp"])
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


# ----------------------------------------------------------------------------- model
def build_model(cfg: dict, n_cls: int, dev, seed: int) -> tuple[nn.Module, nn.Module]:
    """ImageNet ResNet-18 embedding net + freshly initialised CosFace head (seeded, so
    Dense-Clean and Dense-Aug start from identical weights)."""
    mcfg, tcfg = cfg["model"], cfg["train"]
    set_seed(seed, tcfg["deterministic"])
    model = build_face_resnet18(mcfg["embedding_dim"], pretrained=True).to(dev)
    head = CosFaceHead(mcfg["embedding_dim"], n_cls, mcfg["scale"], mcfg["margin"]).to(dev)
    if tcfg["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    return model, head
