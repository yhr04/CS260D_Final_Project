"""Recompute the released calibration ablation's LFW tables from per-pair scores (NumPy only)."""
import csv
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np


RESULTS = Path(__file__).resolve().parents[1] / "results" / "pruning"
STUDY = RESULTS / "calibration_selection"
REFERENCE = RESULTS / "one_shot_seed42"
STRATEGIES = ("random", "coreset", "kcenter", "greedi", "greedy_merge", "sieve")
SEEDS = (42, 43, 44)
BUDGETS = (256, 512, 1024)
METHODS = ("alpha_weighted", "condition_max")
REGIONS = ("eyes", "mouth_nose", "upper_half", "lower_half")
CONDITIONS = ("clean", "square10", "square30", "square50", *REGIONS)
METRICS = (*CONDITIONS, "seen_square_mean", "region_mean")


def verify_artifacts():
    protocol = json.loads((STUDY / "protocol.json").read_text())
    artifacts = protocol["public_artifacts_sha256"]
    expected = {"pair_similarities.npz", "paired_seed_metrics.csv", "coverage_and_timing.csv",
                "held_out_region_accuracy.png", "selection_orders.npz"}
    if set(artifacts) != expected:
        raise ValueError("Incomplete or unexpected artifact manifest")
    for name, expected_digest in artifacts.items():
        digest = hashlib.sha256((STUDY / name).read_bytes()).hexdigest()
        if digest != expected_digest:
            raise ValueError(f"SHA-256 mismatch for {name}")


def fold_counts(scores, same, folds):
    """Choose each threshold using only the other nine folds and count correct held-out pairs."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != same.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid pair scores")
    correct = []
    for k in range(10):
        test = folds == k
        train = ~test
        positive = np.sort(scores[train & same])
        negative = np.sort(scores[train & ~same])
        distinct = np.unique(scores[train])
        # Midpoints avoid splitting tied scores. Match the first maximum when accuracy ties.
        middle = (distinct[1:] + distinct[:-1]) / 2
        cuts = np.r_[distinct[0] - 0.5e-6, middle, distinct[-1] + 0.5e-6]
        hits = positive.size - positive.searchsorted(cuts, side="right") + negative.searchsorted(cuts, side="right")
        threshold = cuts[hits.argmax()]
        correct.append(np.count_nonzero((scores[test] > threshold) == same[test]))
    return np.array(correct)


def accuracy_pct(counts):
    by_condition = dict(zip(CONDITIONS, counts.sum(axis=1) / 6000 * 100))
    by_condition["seen_square_mean"] = (by_condition["square10"] + by_condition["square30"]) / 2
    by_condition["region_mean"] = np.mean([by_condition[c] for c in REGIONS])
    return by_condition


def verify_scores():
    measured = {}
    reference_controls = {}
    with np.load(STUDY / "pair_similarities.npz", allow_pickle=False) as study, np.load(
        REFERENCE / "pair_similarities.npz", allow_pickle=False
    ) as reference:
        same = reference["lfw/issame"]
        folds = reference["lfw/fold"]
        np.testing.assert_array_equal(folds, np.repeat(np.arange(10), 600))
        if same.dtype != np.bool_ or same.shape != (6000,) or int(same.sum()) != 3000:
            raise ValueError("Unexpected LFW pair labels")
        np.testing.assert_array_equal(study["conditions"], CONDITIONS)
        expected_keys = {"conditions"}
        for case in product(STRATEGIES, SEEDS, BUDGETS, METHODS):
            strategy, seed, budget, method = case
            key = f"{strategy}/{seed}/{budget}/{method}"
            if strategy == "random" and budget == 1024:
                scores = np.stack([reference[f"lfw/{seed}/{method}/{c}"] for c in CONDITIONS])
            else:
                scores = study[f"scores/{key}"]
                expected_keys.add(f"scores/{key}")
            if scores.shape != (len(CONDITIONS), len(same)):
                raise ValueError(f"Invalid score matrix for {key}")
            counts = np.stack([fold_counts(row, same, folds) for row in scores])
            expected_keys.add(f"fold_correct/{key}")
            np.testing.assert_array_equal(counts, study[f"fold_correct/{key}"], err_msg=key)
            measured[case] = accuracy_pct(counts)
        if set(study.files) != expected_keys or len(study.files) != len(expected_keys):
            raise ValueError("Unexpected score archive contents")
        for method in ("dense", "magnitude"):
            counts = np.stack([fold_counts(reference[f"lfw/42/{method}/{c}"], same, folds) for c in CONDITIONS])
            reference_controls[method] = accuracy_pct(counts)
    return measured, reference_controls


def verify_tables(measured, controls):
    with (REFERENCE / "metrics.csv").open() as f:
        control_rows = {
            (r["method"], r["condition"]): float(r["accuracy"]) * 100
            for r in csv.DictReader(f) if r["dataset"] == "lfw" and r["method"] in controls and r["scoring_seed"] == "42"
        }
    for method, values in controls.items():
        for metric in CONDITIONS:
            np.testing.assert_allclose(values[metric], control_rows[method, metric], rtol=0, atol=1e-10)

    comparisons = {"minus_random_pp": "random", "minus_full_facility_pp": "coreset", "minus_kcenter_pp": "kcenter"}
    expected = {(case, metric) for case in measured for metric in METRICS}
    with (STUDY / "paired_seed_metrics.csv").open() as f:
        for row in csv.DictReader(f):
            case = (row["strategy"], int(row["seed"]), int(row["budget"]), row["method"])
            metric = row["metric"]
            if (case, metric) not in expected:
                raise ValueError(f"Unexpected or duplicated row: {case}, {metric}")
            expected.remove((case, metric))
            value = measured[case][metric]
            np.testing.assert_allclose(value, float(row["accuracy_pct"]), rtol=0, atol=1e-10, err_msg=str((case, metric)))
            for column, strategy in comparisons.items():
                delta = value - measured[(strategy, *case[1:])][metric]
                np.testing.assert_allclose(delta, float(row[column]), rtol=0, atol=1e-10, err_msg=str((case, metric, column)))
    if expected:
        raise ValueError(f"Missing {len(expected)} measured rows")


if __name__ == "__main__":
    verify_artifacts()
    measurements, references = verify_scores()
    verify_tables(measurements, references)
    print("Verified five artifact hashes, 108 method settings, 8,640 fold counts, "
          "1,080 metric rows and their paired differences.")
    for label, values in references.items():
        print(f"{label}: four-region mean {values['region_mean']:.2f}% | clean {values['clean']:.2f}% | "
              f"square50 {values['square50']:.2f}%")
