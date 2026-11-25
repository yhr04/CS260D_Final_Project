"""Shared helpers: paths, seeding, devices, timing, config and JSON I/O."""
from __future__ import annotations

import copy
import json
import logging
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- paths
def resolve(path: str | os.PathLike) -> Path:
    """Resolve a path relative to the project root (absolute paths are kept)."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_dir(path: str | os.PathLike) -> Path:
    p = resolve(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- config
def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | os.PathLike = "configs/default.yaml", overrides: dict | None = None) -> dict:
    with open(resolve(path), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = _deep_update(copy.deepcopy(cfg), overrides)
    return cfg


# --------------------------------------------------------------------------- seeding / device
def set_seed(seed: int, deterministic: bool = True) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(pref: str = "auto"):
    import torch

    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


def device_sync(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class Timer:
    """Wall-clock timer that synchronises the accelerator before reading the clock."""

    def __init__(self, device=None):
        self.device = device
        self.elapsed = 0.0

    def __enter__(self):
        if self.device is not None:
            device_sync(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.device is not None:
            device_sync(self.device)
        self.elapsed = time.perf_counter() - self._t0
        return False


def env_info() -> dict:
    """Hardware / software versions, recorded with every result file."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
        import torchvision

        info.update(torch=torch.__version__, torchvision=torchvision.__version__,
                    cuda_version=torch.version.cuda, torch_threads=torch.get_num_threads())
        cuda = torch.cuda.is_available()
        info["cuda_available"] = cuda
        try:  # device queries can fail even when is_available() is True (e.g. hidden devices)
            props = torch.cuda.get_device_properties(0) if cuda else None
            info.update(cudnn_version=torch.backends.cudnn.version() if cuda else None,
                        gpu=props.name if props else None,
                        gpu_memory_gb=round(props.total_memory / 1024**3, 2) if props else None)
        except (AssertionError, RuntimeError) as e:
            info.update(cudnn_version=None, gpu=None, gpu_memory_gb=None, gpu_error=str(e))
    except ImportError:
        info["torch"] = None
    for mod in ("numpy", "sklearn", "pandas", "matplotlib"):
        try:
            info[mod] = __import__(mod).__version__
        except ImportError:
            info[mod] = None
    return info


# --------------------------------------------------------------------------- I/O
def _to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


def save_json(obj, path: str | os.PathLike) -> Path:
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, indent=2, ensure_ascii=False)
    return p


def load_json(path: str | os.PathLike):
    with open(resolve(path), "r", encoding="utf-8") as f:
        return json.load(f)


def get_logger(name: str, log_file: str | os.PathLike | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        p = resolve(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(p, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
