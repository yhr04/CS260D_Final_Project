"""Main-tier data: CASIA-WebFace (training) and the verification-pair test sets
(LFW, AgeDB-30, CALFW, CPLFW), all 112x112 aligned RGB.

Both Kaggle archives are read directly from their zip files and decoded once into
uint8 caches under data/cache (git-ignored):

    casia_112.npy            (N, 112, 112, 3) uint8 memmap, ~18.5 GB, sorted by identity
    casia_labels.npy         (N,) int32 identity index (0..10571, folder id_<k> -> k)
    pairs_<name>.npz         images (12000, 112, 112, 3) uint8, issame (6000,) bool

Archive layout (checked 2026-11-21):
    webface.zip: webface_112x112/id_<k>/<k>_<j>.jpg          490,623 images, 10,572 identities
    val.zip:     val/<name>_112x112/<i>.bmp + val/<name>_ann.txt
                 ann line t: "<issame> <name>_112x112/<2t>.bmp <name>_112x112/<2t+1>.bmp";
                 6000 pairs = 10 consecutive folds of 600 (300 positive, then 300 negative).
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .utils import ensure_dir, resolve

H = W = 112
_ID_RE = re.compile(r"id_(\d+)/(\d+)_(\d+)\.jpg$")


# =============================================================================
# CASIA-WebFace cache
# =============================================================================
def casia_paths(cache_dir) -> dict:
    d = resolve(cache_dir)
    return {"images": d / "casia_112.npy", "labels": d / "casia_labels.npy", "meta": d / "casia_meta.json"}


def _list_webface(zip_path) -> tuple[list[str], np.ndarray]:
    with zipfile.ZipFile(resolve(zip_path)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".jpg")]
    keyed = []
    for n in names:
        m = _ID_RE.search(n)
        if m is None:
            raise ValueError(f"unexpected entry {n}")
        keyed.append((int(m.group(1)), int(m.group(3)), n))
    keyed.sort()
    return [k[2] for k in keyed], np.array([k[0] for k in keyed], dtype=np.int32)


def _decode_chunk(args) -> int:
    """Worker: decode zip entries [start, start+len(names)) into the memmap."""
    from PIL import Image

    zip_path, out_path, start, names = args
    out = np.load(out_path, mmap_mode="r+")
    with zipfile.ZipFile(zip_path) as z:
        for i, n in enumerate(names):
            im = Image.open(io.BytesIO(z.read(n))).convert("RGB")
            a = np.asarray(im)
            if a.shape != (H, W, 3):
                raise ValueError(f"{n}: shape {a.shape}")
            out[start + i] = a
    out.flush()
    del out
    return len(names)


def build_casia_cache(zip_path, cache_dir, workers: int = 12, chunk: int = 4000, verbose: bool = True) -> dict:
    """Decode every CASIA image once into a uint8 .npy memmap (idempotent)."""
    p = casia_paths(cache_dir)
    if p["meta"].exists():
        meta = json.loads(p["meta"].read_text(encoding="utf-8"))
        if verbose:
            print(f"[casia] cache exists: {meta['n_images']} images, {meta['n_identities']} identities -> {p['images']}")
        return meta
    ensure_dir(cache_dir)
    names, labels = _list_webface(zip_path)
    n = len(names)
    if verbose:
        print(f"[casia] decoding {n} images ({len(np.unique(labels))} identities) with {workers} processes "
              f"-> {p['images']} ({n * H * W * 3 / 1e9:.1f} GB)")
    arr = np.lib.format.open_memmap(p["images"], mode="w+", dtype=np.uint8, shape=(n, H, W, 3))
    del arr
    jobs = [(str(resolve(zip_path)), str(p["images"]), s, names[s : s + chunk]) for s in range(0, n, chunk)]
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k in ex.map(_decode_chunk, jobs):
            done += k
            if verbose and (done // chunk) % 20 == 0:
                print(f"  {done}/{n}", flush=True)
    # identity id_<k> -> label k; ids are contiguous 0..K-1 in this archive, check it
    uniq = np.unique(labels)
    if not np.array_equal(uniq, np.arange(len(uniq))):
        labels = np.searchsorted(uniq, labels).astype(np.int32)
    np.save(p["labels"], labels)
    counts = np.bincount(labels)
    meta = {
        "source": str(zip_path),
        "n_images": int(n),
        "n_identities": int(len(uniq)),
        "shape_hwc": [H, W, 3],
        "images_per_identity": {"min": int(counts.min()), "median": float(np.median(counts)),
                                "mean": float(counts.mean()), "max": int(counts.max())},
        "first_entry": names[0],
        "last_entry": names[-1],
    }
    p["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if verbose:
        print(f"[casia] done: {meta}")
    return meta


def load_casia(cache_dir) -> tuple[np.ndarray, np.ndarray]:
    """(images memmap (N,112,112,3) uint8 read-only, labels (N,) int64)."""
    p = casia_paths(cache_dir)
    return np.load(p["images"], mmap_mode="r"), np.load(p["labels"]).astype(np.int64)


# =============================================================================
# held-out identities -> validation pairs (model selection, never LFW)
# =============================================================================
def casia_split_path(seed: int):
    return resolve(f"results/splits/casia_split_seed{seed}.npz")


def make_casia_split(labels: np.ndarray, frac: float, min_images: int, pairs_per_class: int, seed: int) -> dict:
    """Hold out `frac` of the identities (among those with >= min_images) and build
    `pairs_per_class` positive + negative verification pairs on their images.

    Pairs are arranged in 10 folds of (pos/10 positive, neg/10 negative), like LFW.
    Returns holdout_ids, val_pairs (P,2) image indices, val_issame (P,), fold (P,).
    """
    rng = np.random.default_rng([seed, 260])
    counts = np.bincount(labels)
    eligible = np.flatnonzero(counts >= min_images)
    n_hold = int(round(frac * len(counts)))
    holdout = np.sort(rng.choice(eligible, size=n_hold, replace=False))
    by_id = {int(c): np.flatnonzero(labels == c) for c in holdout}

    pos, seen = [], set()
    while len(pos) < pairs_per_class:
        c = int(rng.choice(holdout))
        a, b = rng.choice(by_id[c], size=2, replace=False)
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            pos.append(key)
    neg = []
    while len(neg) < pairs_per_class:
        c1, c2 = rng.choice(holdout, size=2, replace=False)
        a, b = int(rng.choice(by_id[int(c1)])), int(rng.choice(by_id[int(c2)]))
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            neg.append(key)
    n_folds = 10
    pf, nf = pairs_per_class // n_folds, pairs_per_class // n_folds
    pairs, issame, fold = [], [], []
    for k in range(n_folds):
        pairs += pos[k * pf : (k + 1) * pf] + neg[k * nf : (k + 1) * nf]
        issame += [True] * pf + [False] * nf
        fold += [k] * (pf + nf)
    return {
        "holdout_ids": holdout.astype(np.int32),
        "val_pairs": np.array(pairs, dtype=np.int64),
        "val_issame": np.array(issame, dtype=bool),
        "fold": np.array(fold, dtype=np.int8),
        "seed": np.array(seed),
    }


def save_casia_split(split: dict, seed: int) -> None:
    p = casia_split_path(seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **split)


def load_casia_split(seed: int) -> dict:
    z = np.load(casia_split_path(seed), allow_pickle=False)
    return {k: z[k] for k in z.files}


def train_indices_and_labels(labels: np.ndarray, holdout_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Training image indices and their labels remapped to 0..C_train-1."""
    keep = ~np.isin(labels, holdout_ids)
    idx = np.flatnonzero(keep)
    train_ids = np.setdiff1d(np.unique(labels), holdout_ids)
    remap = np.full(labels.max() + 1, -1, dtype=np.int64)
    remap[train_ids] = np.arange(len(train_ids))
    return idx, remap[labels], len(train_ids)


def check_casia_split(split: dict, labels: np.ndarray) -> None:
    hold = set(split["holdout_ids"].tolist())
    pl = labels[split["val_pairs"]]
    assert all(int(c) in hold for c in pl.ravel()), "validation pair uses a training identity"
    assert np.array_equal(pl[:, 0] == pl[:, 1], split["val_issame"]), "issame does not match identities"
    assert (split["val_pairs"][:, 0] != split["val_pairs"][:, 1]).all()
    assert len({tuple(p) for p in split["val_pairs"].tolist()}) == len(split["val_pairs"]), "duplicate pairs"


# =============================================================================
# verification test sets (LFW etc.)
# =============================================================================
def pairs_cache_path(cache_dir, name: str):
    return resolve(cache_dir) / f"pairs_{name}.npz"


def build_pairs_cache(zip_path, cache_dir, name: str) -> dict:
    p = pairs_cache_path(cache_dir, name)
    if p.exists():
        return load_pairs(cache_dir, name)
    from PIL import Image

    ensure_dir(cache_dir)
    with zipfile.ZipFile(resolve(zip_path)) as z:
        lines = z.read(f"val/{name}_ann.txt").decode().split()
        rows = [lines[i : i + 3] for i in range(0, len(lines), 3)]
        issame = np.array([r[0] == "1" for r in rows], dtype=bool)
        files = []
        for t, (_, a, b) in enumerate(rows):
            # images are stored in pair order: pair t = (2t, 2t+1)
            assert a.endswith(f"/{2 * t}.bmp") and b.endswith(f"/{2 * t + 1}.bmp"), (t, a, b)
            files += [a, b]
        imgs = np.stack([np.asarray(Image.open(io.BytesIO(z.read("val/" + f))).convert("RGB")) for f in files])
    assert imgs.shape[1:] == (H, W, 3)
    np.savez(p, images=imgs, issame=issame)
    return {"images": imgs, "issame": issame}


def load_pairs(cache_dir, name: str) -> dict:
    z = np.load(pairs_cache_path(cache_dir, name), allow_pickle=False)
    return {"images": z["images"], "issame": z["issame"]}
