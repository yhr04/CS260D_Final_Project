"""LFW-funneled loading, identity-stratified splitting and split persistence.

Images are kept in memory as uint8 arrays of shape (N, H, W, 3). The dataset is
small (~3k images at 128x128), so no DataLoader / worker processes are needed,
which also avoids Windows multiprocessing issues.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .utils import ensure_dir, load_json, resolve, save_json


# --------------------------------------------------------------------------- loading
def _cache_name(dcfg: dict) -> str:
    y0, y1, x0, x1 = dcfg["crop"]
    tag = (
        f"lfw_f{int(dcfg['funneled'])}_min{dcfg['min_faces_per_person']}"
        f"_crop{y0}-{y1}-{x0}-{x1}_r{dcfg['resize']}"
    )
    return tag + ".npz"


def load_lfw(dcfg: dict, verbose: bool = True) -> dict:
    """Load LFW via scikit-learn (downloads once) and cache a uint8 copy.

    Returns dict with keys: images (N,H,W,3 uint8), labels (N, int64),
    target_names (list[str]).
    """
    data_dir = ensure_dir(dcfg["data_dir"])
    cache = data_dir / _cache_name(dcfg)
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        out = {
            "images": z["images"],
            "labels": z["labels"].astype(np.int64),
            "target_names": [str(s) for s in z["target_names"]],
        }
        if verbose:
            print(f"[data] loaded cached arrays from {cache}")
        return out

    from sklearn.datasets import fetch_lfw_people

    y0, y1, x0, x1 = dcfg["crop"]
    if verbose:
        print("[data] fetching LFW with scikit-learn (first run downloads ~230 MB) ...")
    lfw = fetch_lfw_people(
        data_home=str(data_dir / "sklearn"),
        funneled=dcfg["funneled"],
        min_faces_per_person=dcfg["min_faces_per_person"],
        resize=dcfg["resize"],
        color=True,
        slice_=(slice(y0, y1), slice(x0, x1)),
        download_if_missing=True,
    )
    imgs = lfw.images.astype(np.float32)
    # sklearn has returned both [0,1] and [0,255] floats across versions.
    if imgs.max() <= 1.5:
        imgs = imgs * 255.0
    images = np.clip(np.rint(imgs), 0, 255).astype(np.uint8)
    labels = lfw.target.astype(np.int64)
    names = [str(s) for s in lfw.target_names]
    np.savez_compressed(cache, images=images, labels=labels, target_names=np.array(names))
    if verbose:
        print(f"[data] cached arrays to {cache}")
    return {"images": images, "labels": labels, "target_names": names}


def dataset_fingerprint(labels: np.ndarray, images: np.ndarray | None = None) -> str:
    h = hashlib.sha1(np.ascontiguousarray(labels).tobytes())
    if images is not None:
        h.update(str(images.shape).encode())
        h.update(np.ascontiguousarray(images[:: max(1, len(images) // 50)]).tobytes())
    return h.hexdigest()[:16]


def class_counts(labels: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    return np.bincount(labels, minlength=num_classes or int(labels.max()) + 1)


# --------------------------------------------------------------------------- splitting
def stratified_split(labels: np.ndarray, fractions=(0.70, 0.15, 0.15), seed: int = 42) -> dict:
    """Per-identity split of ORIGINAL image indices into train / val / test.

    For every identity with n images: n_val = max(1, round(f_val * n)),
    n_test = max(1, round(f_test * n)), the rest go to train. This guarantees
    every identity appears in all three subsets (requires n >= 3).
    """
    labels = np.asarray(labels)
    f_train, f_val, f_test = fractions
    assert abs(f_train + f_val + f_test - 1.0) < 1e-6, "fractions must sum to 1"
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        n = len(idx)
        if n < 3:
            raise ValueError(f"class {c} has only {n} images; need >= 3 for a 3-way split")
        idx = rng.permutation(idx)
        n_val = max(1, int(round(f_val * n)))
        n_test = max(1, int(round(f_test * n)))
        if n - n_val - n_test < 1:
            raise ValueError(f"class {c}: not enough images left for training")
        va.append(idx[:n_val])
        te.append(idx[n_val : n_val + n_test])
        tr.append(idx[n_val + n_test :])
    return {
        "train": np.sort(np.concatenate(tr)),
        "val": np.sort(np.concatenate(va)),
        "test": np.sort(np.concatenate(te)),
    }


def check_split(split: dict, labels: np.ndarray) -> None:
    """Quality checks: disjoint original images, full coverage, every class everywhere."""
    s = {k: set(map(int, v)) for k, v in split.items()}
    assert not (s["train"] & s["val"]), "train/val overlap"
    assert not (s["train"] & s["test"]), "train/test overlap"
    assert not (s["val"] & s["test"]), "val/test overlap"
    assert len(s["train"] | s["val"] | s["test"]) == len(labels), "split does not cover all images"
    all_classes = set(np.unique(labels).tolist())
    for k, v in split.items():
        missing = all_classes - set(np.unique(labels[v]).tolist())
        assert not missing, f"{k} split misses classes {sorted(missing)[:5]}"


def split_path(seed: int):
    return resolve(f"results/splits/split_seed{seed}.json")


def save_split(split: dict, labels: np.ndarray, seed: int, fractions) -> None:
    save_json(
        {
            "seed": seed,
            "fractions": list(fractions),
            "n_total": int(len(labels)),
            "label_fingerprint": dataset_fingerprint(labels),
            **{k: np.asarray(v).tolist() for k, v in split.items()},
        },
        split_path(seed),
    )


def load_split(seed: int, labels: np.ndarray | None = None) -> dict:
    d = load_json(split_path(seed))
    if labels is not None:
        assert d["n_total"] == len(labels), "saved split was made for a different dataset size"
        assert d["label_fingerprint"] == dataset_fingerprint(labels), "label fingerprint mismatch"
    return {k: np.asarray(d[k], dtype=np.int64) for k in ("train", "val", "test")}
