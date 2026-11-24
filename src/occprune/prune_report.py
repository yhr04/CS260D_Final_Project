"""Tables and analyses over prune_run output directories (notebook 03 / 04).

    from occprune.prune_report import load_runs, metrics_table, summary_table, paired_bootstrap, bank_agreement
    runs = load_runs("results/pruning/runs/*")
    df = metrics_table(runs)                     # long format, one row per seed/method/sparsity/stage/set/condition
    summary_table(df, dataset="lfw")             # mean ± sd over scoring seeds, + region mean
    paired_bootstrap(run_dir, "oagp_max", "oagp", sparsity=0.5)   # per-pair CI on one run
    bank_agreement(run_dir, sparsities=(0.5, 0.7))                 # E4: clean vs occluded importance
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .faceverif import verification_metrics
from .pruning import layerwise_masks, mean_banks
from .utils import resolve

REGIONS = ("eyes", "mouth_nose", "upper_half", "lower_half")
CONDITION_ORDER = ("clean", "square10", "square30", "square50", *REGIONS)


def _sparsity_key(s: float) -> str:
    return f"s{float(s):g}"


def load_runs(pattern) -> dict[str, dict]:
    """{run directory: result.json} for every directory matching `pattern` (str glob or list)."""
    paths = pattern if isinstance(pattern, (list, tuple)) else sorted(glob.glob(str(resolve(pattern))))
    runs = {}
    for p in paths:
        f = Path(p) / "result.json"
        if f.is_file():
            runs[str(p)] = json.loads(f.read_text(encoding="utf-8"))
    if not runs:
        raise FileNotFoundError(f"No prune_run results under {pattern}")
    return runs


def metrics_table(runs: dict[str, dict]) -> pd.DataFrame:
    rows = []

    def add(run, seed, method, sparsity, stage, entry):
        for dataset, conds in entry["metrics"].items():
            for cond, m in conds.items():
                rows.append({"run": Path(run).name, "scoring_seed": seed, "method": method, "sparsity": sparsity,
                             "stage": stage, "dataset": dataset, "condition": cond, "acc": m["acc"],
                             "acc_std_folds": m["acc_std"], "auc": m["auc"], "tar_far1e-3": m["tar_at_far1e-3"],
                             "nonzero_prunable": entry["parameters"]["prunable_nonzero"]})

    for run, r in runs.items():
        seed = r["scoring_seed"]
        add(run, seed, "dense", 0.0, "dense", r["dense"])
        for method, me in r["methods"].items():
            for key, e in me["sparsities"].items():
                sp = float(key[1:])
                for stage in ("one_shot", "fine_tuned"):
                    if stage in e:
                        add(run, seed, method, sp, stage, e[stage])
    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame, dataset: str = "lfw", stage: str = "one_shot", pct: bool = True) -> pd.DataFrame:
    """Accuracy per method x sparsity, averaged over scoring seeds (mean and sd across seeds),
    with the four-region mean. The dense row is included for reference."""
    d = df[(df.dataset == dataset) & ((df.stage == stage) | (df.stage == "dense"))].copy()
    region = (d[d.condition.isin(REGIONS)].groupby(["scoring_seed", "method", "sparsity", "stage"], as_index=False)
              .acc.mean().assign(condition="region_mean"))
    d = pd.concat([d, region], ignore_index=True)
    g = d.groupby(["method", "sparsity", "condition"]).acc
    out = pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1), "n_seeds": g.count()}).reset_index()
    if pct:
        out[["mean", "sd"]] *= 100
    wide = out.pivot_table(index=["method", "sparsity"], columns="condition", values="mean")
    cols = [c for c in (*CONDITION_ORDER, "region_mean") if c in wide.columns]
    return wide[cols]


def paired_bootstrap(run_dir, method_a: str, method_b: str, *, sparsity: float, stage: str = "one_shot",
                     dataset: str = "lfw", conditions=CONDITION_ORDER, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """acc(B) - acc(A) with a 95% CI over resampled pairs, each model using its own fold thresholds.
    method 'dense' refers to the unpruned model."""
    key = _sparsity_key(sparsity)
    with np.load(Path(resolve(run_dir)) / "pair_similarities.npz") as z:
        same, folds = z[f"{dataset}/issame"], z[f"{dataset}/fold"]

        def sims(m, c):
            return z[f"dense/{dataset}/{c}"] if m == "dense" else z[f"{m}/{key}/{stage}/{dataset}/{c}"]

        rng = np.random.default_rng(seed)
        boot_idx = rng.integers(0, len(same), size=(n_boot, len(same)))
        rows = []
        for c in conditions:
            if f"dense/{dataset}/{c}" not in z.files:
                continue
            ca = verification_metrics(sims(method_a, c), same, folds)["correct"].astype(float)
            cb = verification_metrics(sims(method_b, c), same, folds)["correct"].astype(float)
            diff = cb - ca
            boots = diff[boot_idx].mean(1)
            rows.append({"condition": c, "diff_pp": 100 * diff.mean(), "ci_low_pp": 100 * np.percentile(boots, 2.5),
                         "ci_high_pp": 100 * np.percentile(boots, 97.5)})
    return pd.DataFrame(rows).set_index("condition")


def load_banks(run_dir) -> dict[str, dict[str, torch.Tensor]]:
    """{bank: {layer: fp32 scores}} from condition_scores.npz (stored as fp16 / per-layer mean)."""
    banks = {}
    with np.load(Path(resolve(run_dir)) / "condition_scores.npz") as z:
        for k in z.files:
            if k.startswith("scale/"):
                continue
            b, layer = k.split("/", 1)
            scale = float(z[f"scale/{k}"])
            banks.setdefault(b, {})[layer] = torch.from_numpy(z[k].astype(np.float32) * scale)
    return banks


def _ranks(x: torch.Tensor) -> torch.Tensor:
    r = torch.empty_like(x, dtype=torch.float64)
    r[torch.argsort(x)] = torch.arange(len(x), dtype=torch.float64)
    return r


def bank_agreement(run_dir, sparsities=(0.5,)) -> pd.DataFrame:
    """E4: per layer, Spearman correlation between clean-Taylor and occluded-Taylor importance
    (B views each) and the Jaccard overlap of the weights each would KEEP at every sparsity.
    Needs condition_scores.npz (saved by default)."""
    banks = load_banks(run_dir)
    clean = mean_banks(banks["clean_A"], banks["clean_B"])
    occl = mean_banks(banks["occ_A"], banks["occ_B"])
    rows = []
    masks = {s: (layerwise_masks(clean, s), layerwise_masks(occl, s)) for s in sparsities}
    for layer in clean:
        c, o = clean[layer].reshape(-1), occl[layer].reshape(-1)
        rc, ro = _ranks(c), _ranks(o)
        rho = float(torch.corrcoef(torch.stack([rc, ro]))[0, 1])
        row = {"layer": layer, "weights": c.numel(), "spearman": rho,
               "occ_over_clean_mean": float(o.mean() / c.mean())}
        for s, (mc, mo) in masks.items():
            a, b = mc[layer].reshape(-1), mo[layer].reshape(-1)
            row[f"jaccard_keep_{s:g}"] = float((a & b).sum() / (a | b).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("layer")
