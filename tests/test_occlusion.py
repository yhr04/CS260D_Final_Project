"""Quality checks for synthetic occlusion and preprocessing (run: python -m pytest -q)."""
import numpy as np
import pytest

from occprune.occlusion import (
    apply_boxes_np,
    boxes_to_mask_np,
    patch_side,
    sample_train_boxes,
)


# ------------------------------------------------------------------ occlusion
@pytest.mark.parametrize("ratio", [0.1, 0.3, 0.5])
def test_patch_area_close_to_ratio(ratio):
    H = W = 112
    side = patch_side(ratio, H, W)
    assert abs(side * side / (H * W) - ratio) < 0.01


def test_train_boxes_probability_and_range():
    H = W = 112
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
