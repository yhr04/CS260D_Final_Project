"""Small end-to-end runner test using a fitted-checkpoint schema and an offline image fixture."""
import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn

from occprune import pipeline
from occprune.faceverif import CosFaceHead, dataset_seed, pair_conditions, verification_metrics
from occprune.prune_run import _checkpoint_split, run_pruning
from occprune.utils import load_config


class FixtureBackbone(nn.Module):
    def __init__(self, embedding_dim, pretrained=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, stride=8)
        self.conv2 = nn.Conv2d(3, 4, 1)
        self.fc = nn.Linear(4, embedding_dim)

    def forward(self, x):
        return self.fc(self.conv2(self.conv1(x).relu()).mean((2, 3)))


def test_paired_pruning_run_and_saved_results(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "build_face_resnet18", FixtureBackbone)
    cfg = load_config(overrides={
        "model": {"embedding_dim": 4}, "occlusion": {"val_ratios": [0.1]},
        "train": {"channels_last": False, "amp": "none", "batch_size": 4, "eval_batch_size": 4},
    })
    images = np.random.default_rng(42).integers(256, size=(20, 112, 112, 3), dtype=np.uint8)
    labels = np.repeat(np.arange(5), 4)
    np.save(tmp_path / "casia_112.npy", images)
    np.save(tmp_path / "casia_labels.npy", labels)
    pairs = np.array([[12, 13], [12, 16], [16, 17], [13, 17]])
    same = np.array([True, False, True, False])
    folds = np.repeat([0, 1], 2)
    split = tmp_path / "split.npz"
    np.savez(split, val_pairs=pairs, val_issame=same, fold=folds, holdout_ids=[3, 4], seed=42)
    cond = pair_conditions(len(pairs), [0.1], [], dataset_seed(cfg["occlusion"]["val_seed"], "casia42"))
    np.savez(tmp_path / "val_seed42_pair_occlusions.npz", square10=cond["square10"])
    ckpt = tmp_path / "dense.pt"
    net, head = FixtureBackbone(4), CosFaceHead(4, 3)
    torch.save({"model": net.state_dict(), "head": head.state_dict(),
                "meta": {"config": cfg, "mode": "aug", "seed": 42, "n_classes": 3,
                         "split_sha256": hashlib.sha256(split.read_bytes()).hexdigest()}}, ckpt)
    dest = tmp_path / "paired"
    kwargs = dict(cache_dir=tmp_path, split_path=split, masks_dir=tmp_path, data_label="fixture",
                  budget=4, ft_epochs=1, device="cpu")
    run_pruning(ckpt, dest, **kwargs)
    result = json.loads((dest / "result.json").read_text())
    with np.load(dest / "calibration.npz") as calibration:
        sources = np.r_[calibration["clean_indices"], calibration["occluded_indices"]]
        assert len(set(sources)) == 4 and max(sources) < 12
    with np.load(dest / "pair_similarities.npz") as saved:
        for method, n_backward in (("magnitude", 0), ("alpha_weighted", 4), ("condition_max", 4)):
            entry = result["methods"][method]
            assert entry["backward_examples"] == n_backward and entry["fine_tuning"]["steps"] == 3
            for stage in ("one_shot", "fine_tuned"):
                counts = entry[stage]["parameters"]["per_layer"]["conv2"]
                assert counts["kept_by_mask"] == counts["nonzero"] == counts["total"] // 2
                for cond in ("clean", "square10"):
                    sims = saved[f"{method}/{stage}/validation/{cond}"]
                    expected = verification_metrics(sims, same, folds)
                    assert entry[stage]["metrics"]["validation"][cond]["acc"] == expected["acc"]
    with pytest.raises(FileExistsError):
        run_pruning(ckpt, dest, **kwargs)


def test_reject_split_with_correct_seed_but_different_identities(tmp_path):
    other = tmp_path / "different.npz"
    np.savez(other, seed=42, holdout_ids=[1, 2])
    with pytest.raises(ValueError, match="repository split"):
        _checkpoint_split(other, {"seed": 42})
    with pytest.raises(ValueError, match="Split hash"):
        _checkpoint_split(other, {"seed": 42, "split_sha256": "incorrect"})
