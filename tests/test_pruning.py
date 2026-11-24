"""Check scorer, mask selection, exclusions and persistence without downloading weights."""
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from occprune.faceverif import CosFaceHead, build_face_resnet18
from occprune.pruning import (
    condition_scores, apply_masks_, enforce_masks_, layerwise_masks,
    magnitude_scores, mean_banks, prunable_layers, random_scores, taylor_scores,
)


class TinyEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2, 1)
        self.conv2 = nn.Conv2d(2, 3, 1)
        self.bn = nn.BatchNorm2d(3)
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(self.bn(self.conv2(self.conv1(x))).mean(dim=(2, 3)))


def test_resnet18_scope_matches_preregistration():
    layers = prunable_layers(build_face_resnet18(pretrained=False))
    assert len(layers) == 19 and sum(layer.weight.numel() for layer in layers.values()) == 11_157_504
    assert "conv1" not in layers and "fc.0" not in layers
    assert {name for name in layers if "downsample" in name} == {
        "layer2.0.downsample.0", "layer3.0.downsample.0", "layer4.0.downsample.0",
    }


def test_taylor_matches_per_image_cosface_and_restores_state():
    torch.manual_seed(3)
    model, head = TinyEmbedding().eval(), CosFaceHead(2, 17, scale=7.).eval()
    x, y = torch.randn(4, 3, 4, 4), torch.tensor([12, 1, 0, 16])
    weight = model.conv2.weight
    signed = []
    for i in range(len(y)):
        logits, _ = head(model(x[i:i+1]), y[i:i+1])
        grad, = torch.autograd.grad(F.cross_entropy(logits, y[i:i+1]), weight)
        signed.append((weight.detach() * grad).detach())
    reference = torch.stack(signed).abs().mean(0)
    assert not torch.allclose(reference, torch.stack(signed).mean(0).abs())

    model.train()
    model.fc.eval()
    head.train()
    weight.requires_grad_(False)
    weight.grad = torch.full_like(weight, 3.)
    head.weight.grad = torch.ones_like(head.weight)
    state = {name: val.clone() for name, val in model.state_dict().items()}
    actual = taylor_scores(model, head, [(x[:1], y[:1]), (x[1:], y[1:])])
    assert set(actual) == {"conv2"}
    torch.testing.assert_close(actual["conv2"], reference)
    assert model.training and model.bn.training and not model.fc.training and head.training
    assert not weight.requires_grad and (weight.grad == 3).all() and (head.weight.grad == 1).all()
    for name, val in model.state_dict().items():
        torch.testing.assert_close(val, state[name], rtol=0, atol=0)


def test_taylor_rejects_empty_or_unmapped_labels():
    model, head = TinyEmbedding(), CosFaceHead(2, 10)
    x, bad_label = torch.randn(1, 3, 4, 4), torch.tensor([-1])
    with pytest.raises(ValueError, match="no images"):
        taylor_scores(model, head, [])
    with pytest.raises(ValueError, match="remapped training"):
        taylor_scores(model, head, [(x, bad_label)])
    assert model.training and head.training


def test_alpha_banks_are_scaled_per_condition_and_layer_before_weighting():
    clean = {"a": torch.tensor([2., 6.]), "b": torch.zeros(2, dtype=torch.float16)}
    occ = {"a": torch.tensor([800., 0.]), "b": torch.tensor([1., 3.])}
    c_copy, o_copy = clean["a"].clone(), occ["a"].clone()
    actual = condition_scores(clean, occ, method="alpha_weighted", alpha=.25)
    torch.testing.assert_close(actual["a"], torch.tensor([.875, 1.125]))
    torch.testing.assert_close(actual["b"], torch.tensor([.125, .375]))
    torch.testing.assert_close(clean["a"], c_copy)
    torch.testing.assert_close(occ["a"], o_copy)
    with pytest.raises(ValueError, match="shapes differ"):
        condition_scores(clean, {**occ, "b": torch.tensor([1.])}, method="condition_max")
    with pytest.raises(ValueError, match="finite"):
        condition_scores(clean, {**occ, "b": torch.tensor([1., float("nan")])}, method="alpha_weighted")


def test_layerwise_count_breaks_ties_by_flat_index():
    scores = {"a": torch.tensor([[3., 3., 0.], [0., 0., 0.]]), "b": torch.ones(4)}
    masks = layerwise_masks(scores, .5)
    assert masks["a"].tolist() == [[True, True, False], [False, False, True]]
    assert masks["b"].tolist() == [False, False, True, True]
    assert all(mask.all() for mask in layerwise_masks(scores, 0.).values())
    assert not any(mask.any() for mask in layerwise_masks(scores, 1.).values())


@pytest.mark.parametrize("n, sparsity, keep", [(7, .3, 4), (10, .8, 2), (10, .9, 1), (3, .5, 1)])
def test_floor_keep_convention_including_exact_decimal_boundaries(n, sparsity, keep):
    mask = layerwise_masks({"layer": torch.ones(n)}, sparsity)["layer"]
    assert mask.tolist() == [False] * (n - keep) + [True] * keep


def test_mask_projection_after_sgd_momentum_and_unchanged_exclusions():
    model = TinyEmbedding()
    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    scores = magnitude_scores(model)
    torch.testing.assert_close(scores["conv2"], original["conv2.weight"].abs())
    assert scores.keys() == {"conv2"}
    masks = layerwise_masks(scores, .5)
    with pytest.raises(ValueError, match="weight shape"):
        apply_masks_(model, {"conv2": torch.ones(3, dtype=torch.bool)})
    device_masks = apply_masks_(model, masks)
    w = model.conv2.weight
    keep = device_masks["conv2"]
    torch.testing.assert_close(w, original["conv2.weight"] * keep)
    for name, p in model.named_parameters():
        if name != "conv2.weight":
            torch.testing.assert_close(original[name], p, atol=0, rtol=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=.01)
    for _ in range(3):
        optimizer.zero_grad()
        w.sum().backward()
        optimizer.step()
        assert w[~keep].count_nonzero() == (~keep).sum()
        enforce_masks_(model, device_masks)
        assert w[~keep].count_nonzero() == 0
    assert (w[keep] != original["conv2.weight"][keep]).all()


def test_max_preserves_a_condition_specific_weight():
    clean = {"layer": torch.tensor([1.0, 0.0, 1.0])}
    occluded = {"layer": torch.tensor([0.0, 1.0, 0.0])}
    alpha = condition_scores(clean, occluded, method="alpha_weighted")
    maximum = condition_scores(clean, occluded, method="condition_max")
    torch.testing.assert_close(alpha["layer"], torch.tensor([.75, 1.5, .75]))
    torch.testing.assert_close(maximum["layer"], torch.tensor([1.5, 3.0, 1.5]))


def test_random_scores_are_seeded_and_cover_prunable_layers():
    model = TinyEmbedding()
    a, b, c = random_scores(model, 1), random_scores(model, 1), random_scores(model, 2)
    assert a.keys() == {"conv2"} and a["conv2"].shape == model.conv2.weight.shape
    torch.testing.assert_close(a["conv2"], b["conv2"])
    assert not torch.equal(a["conv2"], c["conv2"])


def test_mean_of_half_banks_equals_taylor_on_the_union():
    torch.manual_seed(0)
    model, head = TinyEmbedding(), CosFaceHead(2, 3)
    x, y = torch.randn(6, 3, 4, 4), torch.tensor([0, 1, 2, 0, 1, 2])
    half_a = taylor_scores(model, head, [(x[:3], y[:3])])
    half_b = taylor_scores(model, head, [(x[3:], y[3:])])
    union = taylor_scores(model, head, [(x, y)])
    torch.testing.assert_close(mean_banks(half_a, half_b)["conv2"], union["conv2"])
    with pytest.raises(ValueError):
        mean_banks(half_a, {"other": half_a["conv2"]})
