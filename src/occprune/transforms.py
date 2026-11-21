"""Tensor preprocessing shared by every method:

uint8 -> float [0,1] -> occlusion (fill) -> optional h-flip -> resize -> ImageNet normalisation.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_tensor_images(images_u8: np.ndarray, device) -> torch.Tensor:
    """(N,H,W,3) uint8 numpy -> (N,3,H,W) uint8 tensor on `device`."""
    return torch.from_numpy(np.ascontiguousarray(images_u8)).permute(0, 3, 1, 2).contiguous().to(device)


def boxes_to_mask(boxes: np.ndarray | torch.Tensor, H: int, W: int, device) -> torch.Tensor:
    """(n,3) squares [y0,x0,side] or (n,4) rectangles [y0,x0,h,w] -> (n,1,H,W) bool
    mask (True = occluded). side / h / w = 0 means no occlusion."""
    b = torch.as_tensor(np.asarray(boxes), dtype=torch.long, device=device)
    y0, x0 = b[:, 0].view(-1, 1, 1), b[:, 1].view(-1, 1, 1)
    h = b[:, 2].view(-1, 1, 1)
    w = b[:, 3].view(-1, 1, 1) if b.shape[1] == 4 else h
    ys = torch.arange(H, device=device).view(1, H, 1)
    xs = torch.arange(W, device=device).view(1, 1, W)
    m = (ys >= y0) & (ys < y0 + h) & (xs >= x0) & (xs < x0 + w)
    return m.unsqueeze(1)


def preprocess(
    x_u8: torch.Tensor,
    boxes: np.ndarray | None = None,
    fill: float = 0.0,
    flip: torch.Tensor | None = None,
    img_size: int | None = None,
) -> torch.Tensor:
    """x_u8: (B,3,H,W) uint8 tensor. Returns a normalised float tensor."""
    x = x_u8.float().div_(255.0)
    B, _, H, W = x.shape
    if boxes is not None:
        m = boxes_to_mask(boxes, H, W, x.device)
        x = x.masked_fill(m, fill)
    if flip is not None:
        f = flip.to(x.device).view(-1, 1, 1, 1)
        x = torch.where(f, x.flip(-1), x)
    if img_size is not None and (H != img_size or W != img_size):
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std
