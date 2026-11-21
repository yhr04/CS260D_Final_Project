"""Quality checks for the main tier (CASIA-WebFace -> LFW verification). CPU only, no data needed."""
import numpy as np
import torch

from occprune.casia import check_casia_split, make_casia_split, train_indices_and_labels
from occprune.faceverif import REGION_RECTS, CosFaceHead, best_threshold, pair_conditions, verification_metrics
from occprune.transforms import boxes_to_mask


def _labels():
    rng = np.random.default_rng(0)
    return np.repeat(np.arange(300), rng.integers(2, 40, size=300))


# ------------------------------------------------------------------ held-out split
def test_casia_split_holdout_disjoint_and_pairs_valid():
    y = _labels()
    s = make_casia_split(y, frac=0.1, min_images=5, pairs_per_class=300, seed=42)
    check_casia_split(s, y)  # pairs only use held-out ids, issame matches, no duplicates
    assert len(s["holdout_ids"]) == 30
    assert (np.bincount(y)[s["holdout_ids"]] >= 5).all()
    # 10 folds, each 30 positive + 30 negative
    for k in range(10):
        f = s["fold"] == k
        assert f.sum() == 60 and s["val_issame"][f].sum() == 30
    idx, remap, n = train_indices_and_labels(y, s["holdout_ids"])
    assert not np.isin(y[idx], s["holdout_ids"]).any()
    assert n == 270 and set(remap[idx].tolist()) == set(range(270))


def test_casia_split_deterministic():
    y = _labels()
    a = make_casia_split(y, 0.1, 5, 300, seed=42)
    b = make_casia_split(y, 0.1, 5, 300, seed=42)
    assert all(np.array_equal(a[k], b[k]) for k in a)


# ------------------------------------------------------------------ verification metric
def test_best_threshold_matches_brute_force():
    rng = np.random.default_rng(1)
    same = rng.random(400) < 0.5
    sims = np.round(rng.normal(same * 0.6, 0.4), 2)   # ties on purpose
    t, acc = best_threshold(sims, same)
    grid = np.unique(np.concatenate([sims - 1e-9, sims + 1e-9]))
    brute = max(((sims > g) == same).mean() for g in grid)
    assert abs(acc - brute) < 1e-12
    assert abs(((sims > t) == same).mean() - acc) < 1e-12


def test_verification_metrics_separable_and_random():
    same = np.repeat([True, False], 300)
    folds = np.tile(np.repeat(np.arange(10), 30), 2)
    m = verification_metrics(np.where(same, 0.8, 0.1), same, folds)
    assert m["acc"] == 1.0 and m["auc"] == 1.0 and m["tar_at_far1e-3"] == 1.0
    rng = np.random.default_rng(0)
    r = verification_metrics(rng.random(600), same, folds)
    assert 0.35 < r["acc"] < 0.65


# ------------------------------------------------------------------ occlusions
def test_rect_masks_and_regions():
    m = boxes_to_mask(np.array([[10, 20, 5, 7]]), 112, 112, "cpu")
    assert m.sum() == 35 and m[0, 0, 10, 20] and not m[0, 0, 15, 20]
    sq = boxes_to_mask(np.array([[0, 0, 4]]), 112, 112, "cpu")   # 3-column squares still work
    assert sq.sum() == 16
    for y0, x0, h, w in REGION_RECTS.values():
        assert 0 <= y0 and 0 <= x0 and y0 + h <= 112 and x0 + w <= 112
    c = pair_conditions(50, [0.1, 0.3], ["eyes"], seed=5)
    c2 = pair_conditions(50, [0.1, 0.3], ["eyes"], seed=5)
    assert c["clean"] is None and set(c) == {"clean", "square10", "square30", "eyes"}
    assert all(np.array_equal(c[k], c2[k]) for k in c if c[k] is not None)
    assert (c["square30"][:, 2] == round(np.sqrt(0.3 * 112 * 112))).all()


# ------------------------------------------------------------------ CosFace
def test_cosface_margin_only_on_target():
    torch.manual_seed(0)
    head = CosFaceHead(8, 5, scale=10.0, margin=0.3)
    emb, y = torch.randn(4, 8), torch.tensor([0, 1, 2, 3])
    logits, cos = head(emb, y)
    expect = 10.0 * cos
    expect[torch.arange(4), y] -= 10.0 * 0.3
    assert torch.allclose(logits, expect, atol=1e-6)
    assert cos.abs().max() <= 1 + 1e-6
