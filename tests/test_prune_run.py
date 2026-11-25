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


def test_bn_recalibration_reestimates_running_mean_without_touching_weights():
    from occprune.prune_run import bn_recal_indices, recalibrate_bn_
    from occprune.transforms import preprocess, to_tensor_images

    cfg = load_config()
    cfg["train"].update(amp="off", channels_last=False, batch_size=4, hflip=False)
    cfg["occlusion"]["train_prob"] = 0.0
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 2, 1, bias=False), nn.BatchNorm2d(2))
    model[1].running_mean.fill_(5.0)
    weight = model[0].weight.detach().clone()
    images = np.random.default_rng(0).integers(0, 256, (12, 112, 112, 3), dtype=np.uint8)
    idx = bn_recal_indices(np.arange(12), 12, seed=0)
    info = recalibrate_bn_(model, images, idx, cfg, 0, torch.device("cpu"))
    with torch.no_grad():
        features = model[0](preprocess(to_tensor_images(images, "cpu")))
    assert info["batches"] == 3 and info["bn_layers"] == 1
    assert torch.allclose(model[1].running_mean, features.mean((0, 2, 3)), atol=1e-5)
    assert torch.equal(model[0].weight, weight) and model[1].momentum == 0.1
    with pytest.raises(ValueError):
        bn_recal_indices(np.arange(12), 13, seed=0)


def _synthetic_run(seed, stages):
    """Minimal result.json: one method / sparsity, LFW regions only, accuracy given per stage."""
    from occprune.faceverif import REGIONS

    def entry(acc):
        return {"metrics": {"lfw": {c: {"acc": acc, "acc_std": 0.0, "auc": acc, "tar_at_far1e-3": acc}
                                    for c in REGIONS}},
                "parameters": {"prunable_nonzero": 1}}
    return {"scoring_seed": seed, "dense": entry(0.95),
            "methods": {"magnitude": {"sparsities": {"s0.6": {s: entry(a) for s, a in stages.items()}}}}}


def test_stage_report_counts_overlapping_one_shot_once_and_reports_gap_closed(tmp_path):
    from occprune.prune_report import stage_table, write_stage_report

    runs = {"bn": _synthetic_run(42, {"one_shot": 0.60, "bn_recal": 0.80}),
            "ft": _synthetic_run(42, {"one_shot": 0.60, "fine_tuned": 0.90})}
    t = stage_table(runs).set_index("stage")
    assert t.loc["one_shot", "n_seeds"] == 1 and t.loc["one_shot", "mean"] == pytest.approx(60)
    assert t.loc["bn_recal", "gap_closed"] == pytest.approx(20 / 30)
    for name, r in runs.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "result.json").write_text(json.dumps(r))
    written = write_stage_report([str(tmp_path / "bn"), str(tmp_path / "ft")], tmp_path / "stages",
                                 methods=("magnitude",))
    assert all(p.is_file() for p in written.values())
    assert "67%" in written["markdown"].read_text(encoding="utf-8")
