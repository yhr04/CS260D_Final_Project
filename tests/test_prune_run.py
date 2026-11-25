"""Small end-to-end runner test using a fitted-checkpoint schema and an offline image fixture."""
import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn

from occprune import pipeline
from occprune.faceverif import CosFaceHead, dataset_seed, pair_conditions, verification_metrics
from occprune.prune_report import (
    bank_agreement, load_banks, load_runs, metrics_table, paired_bootstrap, summary_table, write_alpha_report,
    write_grid_report,
)
from occprune.protocol import verify_checkpoint_split
from occprune.prune_run import METHODS, run_pruning
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
        "data": {"num_workers": 0},
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
                  budget=4, sparsities=(0.5, 0.75), ft_epochs=1, ft_lr=0.05, device="cpu")
    run_pruning(ckpt, dest, **kwargs)
    result = json.loads((dest / "result.json").read_text())
    assert result["methods_run"] == list(METHODS)
    with np.load(dest / "calibration.npz") as calibration:
        a, b = calibration["half_A_indices"], calibration["half_B_indices"]
        assert len(set(np.r_[a, b])) == 4 and max(np.r_[a, b]) < 12
        assert calibration["half_A_boxes"].shape == calibration["half_B_boxes"].shape == (2, 3)
    expected_backward = {"random": 0, "magnitude": 0, "clean_taylor": 4, "occ_taylor": 4,
                         "mixed_taylor": 4, "oagp": 4, "oagp_max": 4}
    with np.load(dest / "pair_similarities.npz") as saved:
        for method, n_backward in expected_backward.items():
            entry = result["methods"][method]
            assert entry["backward_examples"] == n_backward
            for key, keep_frac in (("s0.5", 0.5), ("s0.75", 0.25)):
                e = entry["sparsities"][key]
                ft = e["fine_tuning"]
                assert ft["steps"] == 3 and ft["lr_first"] == 0.05 and ft["lr_last"] < 0.05
                for stage in ("one_shot", "fine_tuned"):
                    counts = e[stage]["parameters"]["per_layer"]["conv2"]
                    assert counts["kept_by_mask"] == counts["nonzero"] == int(keep_frac * counts["total"])
                    for cond in ("clean", "square10"):
                        sims = saved[f"{method}/{key}/{stage}/validation/{cond}"]
                        expected = verification_metrics(sims, same, folds)
                        assert e[stage]["metrics"]["validation"][cond]["acc"] == expected["acc"]

    # aggregation helpers read the same directory
    df = metrics_table(load_runs([str(dest)]))
    assert set(df.method) == {"dense", *METHODS}
    table = summary_table(df, dataset="validation")
    assert ("oagp_max", 0.75) in table.index and "clean" in table.columns
    boot = paired_bootstrap(dest, "oagp", "oagp_max", sparsity=0.5, dataset="validation", n_boot=50)
    assert list(boot.index) == ["clean", "square10"]
    banks = load_banks(dest)
    assert set(banks) == {"clean_A", "clean_B", "occ_A", "occ_B"}
    agree = bank_agreement(dest, sparsities=(0.5,))
    assert list(agree.index) == ["conv2"] and -1 <= agree.spearman.iloc[0] <= 1
    written = write_grid_report([str(dest)], tmp_path / "report", bank_sparsities=(0.5,))
    for key in ("metrics_long", "validation_one_shot", "validation_fine_tuned", "bank_agreement"):
        assert written[key].is_file()
    assert (tmp_path / "report" / "sparsity_validation_one_shot.png").is_file()
    alpha = write_alpha_report([str(dest)], tmp_path / "alpha")
    assert all(p.is_file() for p in alpha.values())
    assert "Selected on validation squares" in alpha["markdown"].read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_pruning(ckpt, dest, **kwargs)


def test_reject_split_with_correct_seed_but_different_identities(tmp_path):
    other = tmp_path / "different.npz"
    np.savez(other, seed=42, holdout_ids=[1, 2])
    with pytest.raises(ValueError, match="repository split"):
        verify_checkpoint_split(other, {"seed": 42})
    with pytest.raises(ValueError, match="Split hash"):
        verify_checkpoint_split(other, {"seed": 42, "split_sha256": "incorrect"})
