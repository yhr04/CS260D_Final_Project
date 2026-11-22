"""Layerwise, unstructured pruning for the shared ResNet-18 + CosFace model."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from fractions import Fraction

import torch
from torch import nn
from torch.nn import functional as F


def prunable_layers(model: nn.Module) -> dict[str, nn.Conv2d]:
    """All backbone convolutions except the first (ResNet's top-level conv1)."""
    layers = {name: layer for name, layer in model.named_modules()
              if isinstance(layer, nn.Conv2d) and name != "conv1"}
    if not layers:
        raise ValueError("Expected convolutional layers beyond the stem.")
    return layers


def magnitude_scores(model: nn.Module) -> dict[str, torch.Tensor]:
    """Absolute weight magnitude per prunable coordinate, without a backward pass."""
    return {name: layer.weight.detach().abs().cpu() for name, layer in prunable_layers(model).items()}


@torch.enable_grad()
def taylor_scores(
    model: nn.Module, head: nn.Module, batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Mean absolute per-image Taylor scores for ONE calibration condition.

    Each batch supplies preprocessed floating-point NCHW images and remapped
    training identity labels. Source selection and occlusion belong to the caller.
    Batch boundaries affect I/O only. Each image gets a separate CosFace backward
    pass with eval-mode BN. Existing gradients and training flags are preserved.
    """
    layers = prunable_layers(model)
    weights = tuple(layer.weight for layer in layers.values())
    device = weights[0].device
    scores = {name: torch.zeros_like(layer.weight, dtype=torch.float32) for name, layer in layers.items()}
    modes = {module: module.training for module in (*model.modules(), *head.modules())}
    requires_grad = [weight.requires_grad for weight in weights]
    n = 0
    try:
        model.eval()
        head.eval()
        for weight in weights:
            weight.requires_grad_(True)
        for x, y in batches:
            if x.ndim != 4 or not x.is_floating_point() or y.shape != (len(x),) or y.dtype != torch.long:
                raise ValueError("Expected floating-point NCHW images and one torch.long target per image.")
            # Reject negative held-out labels before passing them into CUDA one_hot.
            if len(y) and (int(y.min()) < 0 or int(y.max()) >= head.weight.shape[0]):
                raise ValueError("Targets must be remapped training identity indices within the CosFace head.")
            x, y = x.to(device), y.to(device)
            for i in range(len(y)):
                target = y[i:i + 1]
                logits, _ = head(model(x[i:i + 1]), target)
                loss = F.cross_entropy(logits, target)
                grads = torch.autograd.grad(loss, weights)
                for total, grad in zip(scores.values(), grads):
                    total.add_(grad.detach().abs())
            n += len(y)
    finally:
        for weight, flag in zip(weights, requires_grad):
            weight.requires_grad_(flag)
        for module, was_training in modes.items():
            module.training = was_training
    if not n:
        raise ValueError("The calibration condition has no images.")

    # The checkpoint is fixed across images, so |w| factors out of the sum.
    result = {}
    for name, weight in zip(scores, weights):
        score = scores[name].div_(n).mul_(weight.detach().abs()).cpu()
        if not torch.isfinite(score).all():
            raise ValueError(f"Non-finite Taylor importance in {name}.")
        result[name] = score
    return result


def _check_scores(name: str, score: torch.Tensor) -> torch.Tensor:
    flat = score.detach().reshape(-1).cpu()
    if not flat.numel() or not flat.is_floating_point() or not torch.isfinite(flat).all() or (flat < 0).any():
        raise ValueError(f"Scores in {name} must be finite, nonnegative, nonempty floating-point tensors.")
    return flat


def condition_scores(
    clean: Mapping[str, torch.Tensor], occluded: Mapping[str, torch.Tensor],
    *, method: str, alpha: float = 0.5, eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Normalize each bank per layer, then take the weighted mean or coordinatewise max."""
    if method not in ("alpha_weighted", "condition_max"):
        raise ValueError(f"Unknown aggregation: {method}")
    if not 0 <= alpha <= 1 or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Require alpha in [0,1] and a finite positive epsilon.")
    if not clean or clean.keys() != occluded.keys():
        raise ValueError("Clean and occluded banks must cover the same layers.")
    scores = {}
    for name, c in clean.items():
        o = occluded[name]
        if c.shape != o.shape:
            raise ValueError(f"Condition score shapes differ in {name}.")
        # At least fp32 so epsilon is representable, including all-zero fp16 banks.
        c_values = _check_scores(name, c).to(torch.promote_types(c.dtype, torch.float32))
        o_values = _check_scores(name, o).to(torch.promote_types(o.dtype, torch.float32))
        c_norm = c_values / (c_values.mean() + eps)
        o_norm = o_values / (o_values.mean() + eps)
        if method == "alpha_weighted":
            combined = (1 - alpha) * c_norm + alpha * o_norm
        else:
            combined = torch.maximum(c_norm, o_norm)
        scores[name] = combined.reshape(c.shape)
    return scores


def layerwise_masks(scores: Mapping[str, torch.Tensor], sparsity: float) -> dict[str, torch.Tensor]:
    """Keep floor((1-sparsity) * numel) per layer. Lower flat indices drop first on ties."""
    if not scores or not 0 <= sparsity <= 1:
        raise ValueError("Require at least one scored layer and sparsity in [0,1].")
    # Parse the configured decimal once: binary 1 - 0.8 is slightly below 0.2.
    keep_fraction = 1 - Fraction(str(sparsity))
    masks = {}
    for name, score in scores.items():
        flat = _check_scores(name, score)
        n_drop = flat.numel() - math.floor(keep_fraction * flat.numel())
        keep = torch.ones(flat.shape, dtype=torch.bool)
        if n_drop:
            keep[torch.argsort(flat, stable=True)[:n_drop]] = False
        masks[name] = keep.reshape(score.shape)
    return masks


@torch.no_grad()
def enforce_masks_(model: nn.Module, device_masks: Mapping[str, torch.Tensor]) -> None:
    """Project weights after EVERY optimizer update with the masks from apply_masks_."""
    for name, layer in prunable_layers(model).items():
        layer.weight.mul_(device_masks[name])


def apply_masks_(model: nn.Module, masks: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Validate all masks before mutating weights. Return masks on their weights' devices."""
    layers = prunable_layers(model)
    if masks.keys() != layers.keys():
        raise ValueError("Masks must cover exactly the prunable backbone convolutions.")
    device_masks = {}
    for name, layer in layers.items():
        mask = masks[name]
        if mask.dtype != torch.bool or mask.shape != layer.weight.shape:
            raise ValueError(f"Expected a boolean mask matching the weight shape in {name}.")
        device_masks[name] = mask.detach().to(layer.weight.device)
    enforce_masks_(model, device_masks)
    return device_masks
