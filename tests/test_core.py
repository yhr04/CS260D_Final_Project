"""Quality checks for splitting and occlusion (run: python -m pytest -q)."""
import numpy as np
import pytest

from occprune.data import check_split, stratified_split
from occprune.occlusion import (
    apply_boxes_np,
    boxes_to_mask_np,
    make_eval_boxes,
    patch_side,
    sample_train_boxes,
)


def _labels():
    rng = np.random.default_rng(0)
    counts = rng.integers(20, 120, size=15)
    return np.repeat(np.arange(15), counts)


# ------------------------------------------------------------------ split
def test_split_disjoint_complete_and_stratified():
    y = _labels()
    s = stratified_split(y, (0.7, 0.15, 0.15), seed=42)
    check_split(s, y)  # disjoint, full coverage, every class in every subset
    for c in np.unique(y):
        n = (y == c).sum()
        assert (y[s["val"]] == c).sum() == max(1, round(0.15 * n))
        assert (y[s["test"]] == c).sum() == max(1, round(0.15 * n))


def test_split_is_deterministic_and_seed_dependent():
    y = _labels()
    a, b = stratified_split(y, seed=42), stratified_split(y, seed=42)
    c = stratified_split(y, seed=43)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["test"], c["test"])


def test_split_rejects_tiny_classes():
    with pytest.raises(ValueError):
        stratified_split(np.array([0, 0, 1, 1, 1, 1]), seed=0)


# ------------------------------------------------------------------ occlusion
@pytest.mark.parametrize("ratio", [0.1, 0.3, 0.5])
def test_patch_area_close_to_ratio(ratio):
    H = W = 128
    side = patch_side(ratio, H, W)
    assert abs(side * side / (H * W) - ratio) < 0.01


def test_eval_boxes_inside_image_deterministic_and_independent():
    H = W = 128
    a = make_eval_boxes(200, [0.0, 0.1, 0.3, 0.5], 3, H, W, seed=1)
    b = make_eval_boxes(200, [0.0, 0.1, 0.3, 0.5], 3, H, W, seed=1)
    c = make_eval_boxes(200, [0.0, 0.1, 0.3, 0.5], 3, H, W, seed=2)
    assert a[0.0].shape == (1, 200, 3) and (a[0.0][..., 2] == 0).all()
    for r in (0.1, 0.3, 0.5):
        assert a[r].shape == (3, 200, 3)
        assert np.array_equal(a[r], b[r])                 # reproducible
        assert not np.array_equal(a[r], c[r])             # val vs test seeds differ
        assert not np.array_equal(a[r][0], a[r][1])       # repeats differ
        y0, x0, s = a[r][..., 0], a[r][..., 1], a[r][..., 2]
        assert (y0 >= 0).all() and (x0 >= 0).all() and (y0 + s <= H).all() and (x0 + s <= W).all()


def test_train_boxes_probability_and_range():
    H = W = 128
    rng = np.random.default_rng(0)
    b = sample_train_boxes(20000, 0.5, (0.1, 0.3), H, W, rng)
    occ = b[:, 2] > 0
    assert abs(occ.mean() - 0.5) < 0.02
    ratios = b[occ, 2] ** 2 / (H * W)
    assert ratios.min() >= 0.09 and ratios.max() <= 0.31
    assert ((b[:, 0] + b[:, 2]) <= H).all() and ((b[:, 1] + b[:, 2]) <= W).all()


def test_mask_zeroes_exactly_the_box_and_keeps_rest():
    rng = np.random.default_rng(0)
    imgs = rng.integers(1, 256, size=(4, 32, 32, 3), dtype=np.uint8)
    boxes = np.array([[0, 0, 5], [10, 3, 8], [0, 0, 0], [24, 24, 8]], dtype=np.int32)
    out = apply_boxes_np(imgs, boxes)
    m = boxes_to_mask_np(boxes, 32, 32)
    assert (out[m] == 0).all()
    assert np.array_equal(out[~m], imgs[~m])
    assert m.reshape(4, -1).sum(1).tolist() == [25, 64, 0, 64]
    assert np.array_equal(out[2], imgs[2])


# ------------------------------------------------------------------ torch preprocessing
def test_torch_mask_matches_numpy_and_labels_untouched():
    torch = pytest.importorskip("torch")
    from occprune.transforms import IMAGENET_MEAN, IMAGENET_STD, preprocess, to_tensor_images

    rng = np.random.default_rng(1)
    imgs = rng.integers(1, 256, size=(3, 40, 40, 3), dtype=np.uint8)
    labels = torch.tensor([5, 7, 9])
    boxes = np.array([[2, 3, 10], [0, 0, 0], [30, 30, 10]], dtype=np.int32)
    x = preprocess(to_tensor_images(imgs, "cpu"), boxes, fill=0.0)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    raw = (x * std + mean).permute(0, 2, 3, 1).numpy()
    m = boxes_to_mask_np(boxes, 40, 40)
    assert np.allclose(raw[m], 0.0, atol=1e-5)
    assert np.allclose(raw[~m], imgs[~m] / 255.0, atol=1e-5)
    assert labels.tolist() == [5, 7, 9]
