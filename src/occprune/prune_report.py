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

from .faceverif import REGIONS, verification_metrics
from .prune_run import sparsity_key
from .pruning import layerwise_masks, mean_banks
from .utils import resolve

CONDITION_ORDER = ("clean", "square10", "square30", "square50", *REGIONS)


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
    key = sparsity_key(sparsity)
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


# =============================================================================
# grid report: tables + figures written to results/
# =============================================================================
# Categorical slots 1-7 of the reference palette (fixed order = method order), plus distinct
# markers so methods stay distinguishable without colour. Dense is a grey reference line.
METHOD_STYLE = {
    "random": ("#2a78d6", "o"), "magnitude": ("#eb6834", "s"), "clean_taylor": ("#1baf7a", "^"),
    "occ_taylor": ("#eda100", "v"), "mixed_taylor": ("#e87ba4", "D"), "oagp": ("#008300", "P"),
    "oagp_max": ("#6250d6", "X"),
}
COND_TITLE = {"clean": "Clean", "square10": "Square 10%", "square30": "Square 30%", "square50": "Square 50%",
              "eyes": "Eyes", "mouth_nose": "Mouth + nose", "upper_half": "Upper half", "lower_half": "Lower half",
              "region_mean": "Mean of four held-out regions"}


def seed_stats(df: pd.DataFrame, dataset: str, stage: str) -> pd.DataFrame:
    """Long table: method, sparsity, condition (incl. region_mean), mean / sd over scoring seeds (%)."""
    d = df[(df.dataset == dataset) & ((df.stage == stage) | (df.stage == "dense"))]
    region = (d[d.condition.isin(REGIONS)].groupby(["scoring_seed", "method", "sparsity", "stage"], as_index=False)
              .acc.mean().assign(condition="region_mean"))
    d = pd.concat([d, region], ignore_index=True)
    g = d.groupby(["method", "sparsity", "condition"]).acc
    return pd.DataFrame({"mean": 100 * g.mean(), "sd": 100 * g.std(ddof=1), "n_seeds": g.count()}).reset_index()


def formatted_table(stats: pd.DataFrame, conditions=(*CONDITION_ORDER, "region_mean")) -> pd.DataFrame:
    """'mean ± sd' strings; rows = Dense-Aug, then method @ sparsity in canonical order."""
    from .prune_run import METHOD_LABEL, METHODS

    cells = [f"{m:.2f} ± {sd:.2f}" if n > 1 else f"{m:.2f}"
             for m, sd, n in zip(stats["mean"], stats["sd"], stats["n_seeds"])]
    wide = stats.assign(cell=cells).pivot_table(index=["method", "sparsity"], columns="condition",
                                                values="cell", aggfunc="first")
    order = [("dense", 0.0)] + [(m, sp) for m in METHODS for sp in sorted(stats.sparsity.unique()) if sp > 0]
    wide = wide.reindex([o for o in order if o in wide.index])[[c for c in conditions if c in wide.columns]]
    wide.index = ["Dense-Aug" if m == "dense" else f"{METHOD_LABEL[m]} @ {int(round(sp * 100))}%"
                  for m, sp in wide.index]
    return wide.rename(columns=COND_TITLE)


def plot_sparsity_curves(stats: pd.DataFrame, conditions=("clean", "square50", "region_mean"), title: str = "",
                         path=None):
    """Accuracy vs sparsity, one panel per condition, mean ± sd band over scoring seeds."""
    import matplotlib.pyplot as plt

    from .prune_run import METHOD_LABEL

    fig, axes = plt.subplots(1, len(conditions), figsize=(4.4 * len(conditions), 3.8), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, cond in zip(axes, conditions):
        c = stats[stats.condition == cond]
        dense = c[c.method == "dense"]["mean"]
        if len(dense):
            ax.axhline(float(dense.iloc[0]), color="#8a8a85", lw=1.2, ls="--", label="Dense-Aug", zorder=1)
        for m, (color, marker) in METHOD_STYLE.items():
            r = c[c.method == m].sort_values("sparsity")
            if not len(r):
                continue
            x = 100 * r.sparsity.to_numpy()
            ax.plot(x, r["mean"], color=color, marker=marker, ms=6, lw=2, label=METHOD_LABEL[m], zorder=3)
            if (r.n_seeds > 1).any():
                sd = r["sd"].fillna(0).to_numpy()
                ax.fill_between(x, r["mean"] - sd, r["mean"] + sd, color=color, alpha=0.15, lw=0, zorder=2)
        ax.set_title(COND_TITLE.get(cond, cond), fontsize=10)
        ax.set_xlabel("layerwise sparsity (%)")
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("verification accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    if path is not None:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig


def _markdown(table: pd.DataFrame) -> str:
    lines = ["| | " + " | ".join(table.columns) + " |", "|---" * (len(table.columns) + 1) + "|"]
    lines += [f"| {i} | " + " | ".join(str(v) for v in row) + " |" for i, row in zip(table.index, table.to_numpy())]
    return "\n".join(lines) + "\n"


def _plot_bank_agreement(a: pd.DataFrame, sparsities, path) -> None:
    import matplotlib.pyplot as plt

    per_layer = a.groupby("layer", sort=False).mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(per_layer))
    ax.plot(x, per_layer["spearman"], color="#2a78d6", marker="o", lw=2, label="Spearman rho of scores")
    for sp, (color, marker) in zip(sparsities, (("#eb6834", "s"), ("#1baf7a", "^"), ("#6250d6", "D"))):
        ax.plot(x, per_layer[f"jaccard_keep_{sp:g}"], color=color, marker=marker, lw=2,
                label=f"Jaccard of kept weights, {int(round(sp * 100))}% sparsity")
    ax.set_xticks(x, [n.replace(".downsample.0", ".ds") for n in per_layer.index], rotation=60, ha="right",
                  fontsize=7.5)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("agreement")
    ax.set_title("Clean-Taylor vs Occ-Taylor importance per layer (mean over scoring seeds)", fontsize=10)
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_grid_report(run_dirs, out_dir, *, bank_sparsities=(0.5, 0.7, 0.9),
                      contrasts=(("clean_taylor", "occ_taylor"), ("oagp", "oagp_max"), ("magnitude", "oagp_max"),
                                 ("magnitude", "occ_taylor"))) -> dict:
    """Write every table / figure for a set of prune_run directories into `out_dir`:
    metrics_long.csv, summary_<dataset>_<stage>.{csv,md}, sparsity_<dataset>_<stage>.png,
    bootstrap_lfw.csv (B - A per seed / stage / sparsity / condition) and bank_agreement.{csv,png}."""
    out = Path(resolve(out_dir))
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(list(run_dirs))
    df = metrics_table(runs)
    df.to_csv(out / "metrics_long.csv", index=False)
    written = {"metrics_long": out / "metrics_long.csv"}
    names = {"lfw": "LFW", "validation": "held-out CASIA pairs"}
    for dataset in sorted(df.dataset.unique()):
        for stage in ("one_shot", "fine_tuned"):
            if not (df.stage == stage).any():
                continue
            stats = seed_stats(df, dataset, stage)
            stats.to_csv(out / f"summary_{dataset}_{stage}.csv", index=False)
            (out / f"summary_{dataset}_{stage}.md").write_text(_markdown(formatted_table(stats)), encoding="utf-8")
            plot_sparsity_curves(stats, path=out / f"sparsity_{dataset}_{stage}.png",
                                 title=f"{names.get(dataset, dataset)}, {stage.replace('_', '-')} "
                                       "(mean ± sd over scoring seeds)")
            written[f"{dataset}_{stage}"] = out / f"summary_{dataset}_{stage}.md"

    rows = []
    for run, r in runs.items():
        if "lfw" not in r["dense"]["metrics"]:
            continue
        for a, b in contrasts:
            if a not in r["methods"] or b not in r["methods"]:
                continue
            for key, entry in r["methods"][a]["sparsities"].items():
                for stage in ("one_shot", "fine_tuned"):
                    if stage not in entry:
                        continue
                    t = paired_bootstrap(run, a, b, sparsity=float(key[1:]), stage=stage)
                    rows += [{"scoring_seed": r["scoring_seed"], "stage": stage, "sparsity": float(key[1:]),
                              "a": a, "b": b, "condition": c, **v} for c, v in t.to_dict("index").items()]
    if rows:
        pd.DataFrame(rows).to_csv(out / "bootstrap_lfw.csv", index=False)
        written["bootstrap"] = out / "bootstrap_lfw.csv"

    agree = [bank_agreement(run, sparsities=bank_sparsities).assign(scoring_seed=r["scoring_seed"])
             for run, r in runs.items()
             if (Path(run) / "condition_scores.npz").is_file() and {"clean_taylor", "occ_taylor"} <= set(r["methods"])]
    if agree:
        a = pd.concat(agree).reset_index()
        a.to_csv(out / "bank_agreement.csv", index=False)
        _plot_bank_agreement(a, bank_sparsities, out / "bank_agreement.png")
        written["bank_agreement"] = out / "bank_agreement.csv"
    return written


# =============================================================================
# alpha sweep (OAGP): choose alpha on held-out CASIA pairs, report LFW
# =============================================================================
ALPHA_SELECT = ("square10", "square30", "square50")   # validation conditions used to pick alpha


def alpha_table(runs: dict[str, dict], method: str = "oagp") -> pd.DataFrame:
    """Long table over runs that contain `method`: alpha, sparsity, metric, mean / sd over scoring seeds (%).
    metric = 'val_clean', 'val_square_mean' (selection criterion) or 'lfw_region_mean' (reported)."""
    frames = []
    for run, r in runs.items():
        if method in r["methods"]:
            d = metrics_table({run: r})
            frames.append(d[(d.method == method) & (d.stage == "one_shot")].assign(alpha=float(r["alpha"])))
    if not frames:
        raise ValueError(f"No runs contain {method}")
    d = pd.concat(frames, ignore_index=True)
    n_alpha = d.alpha.nunique()
    shared = d.groupby("sparsity").alpha.nunique()
    d = d[d.sparsity.isin(shared[shared == n_alpha].index)]   # only sparsities every alpha was run at
    keys = ["alpha", "sparsity", "scoring_seed"]
    parts = {
        "val_clean": d[(d.dataset == "validation") & (d.condition == "clean")],
        "val_square_mean": d[(d.dataset == "validation") & d.condition.isin(ALPHA_SELECT)],
        "lfw_region_mean": d[(d.dataset == "lfw") & d.condition.isin(REGIONS)],
    }
    out = []
    for metric, p in parts.items():
        if len(p):
            g = p.groupby(keys).acc.mean().groupby(["alpha", "sparsity"])
            out.append(pd.DataFrame({"mean": 100 * g.mean(), "sd": 100 * g.std(ddof=1), "n_seeds": g.count()})
                       .reset_index().assign(metric=metric))
    return pd.concat(out, ignore_index=True)


def write_alpha_report(run_dirs, out_dir, method: str = "oagp") -> dict:
    """alpha_sweep.csv, alpha_sweep.md (with the validation-selected alpha per sparsity) and alpha_sweep.png."""
    import matplotlib.pyplot as plt

    out = Path(resolve(out_dir))
    out.mkdir(parents=True, exist_ok=True)
    t = alpha_table(load_runs(list(run_dirs)), method)
    t.to_csv(out / "alpha_sweep.csv", index=False)

    titles = {"val_clean": "Validation clean", "val_square_mean": "Validation squares (selection)",
              "lfw_region_mean": "LFW four-region mean"}
    sparsities = sorted(t.sparsity.unique())
    lines = []
    for metric in [m for m in titles if m in set(t.metric)]:
        w = t[t.metric == metric].assign(cell=lambda x: [f"{m:.2f} ± {s:.2f}" for m, s in zip(x["mean"], x["sd"])])
        wide = w.pivot_table(index="alpha", columns="sparsity", values="cell", aggfunc="first")
        wide.columns = [f"{int(round(100 * s))}%" for s in wide.columns]
        wide.index = [f"α = {a:g}" for a in wide.index]
        lines += [f"### {titles[metric]}", "", _markdown(wide)]
    sel = t[t.metric == "val_square_mean"]
    if len(sel):
        best = sel.loc[sel.groupby("sparsity")["mean"].idxmax()]
        lines += ["### Selected on validation squares", "",
                  *[f"- {int(round(100 * s))}% sparsity: α = {a:g}" for s, a in zip(best.sparsity, best.alpha)], ""]
    (out / "alpha_sweep.md").write_text("\n".join(lines), encoding="utf-8")

    panels = [m for m in ("val_square_mean", "lfw_region_mean") if m in set(t.metric)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.6))
    axes = np.atleast_1d(axes)
    blues = ("#9cc0ee", "#2a78d6", "#0e4590", "#06264f")   # sequential: light = low sparsity
    for ax, metric in zip(axes, panels):
        for sp, color in zip(sparsities, blues):
            r = t[(t.metric == metric) & (t.sparsity == sp)].sort_values("alpha")
            ax.plot(r.alpha, r["mean"], color=color, marker="o", ms=6, lw=2, label=f"{int(round(100 * sp))}%")
            ax.fill_between(r.alpha, r["mean"] - r["sd"].fillna(0), r["mean"] + r["sd"].fillna(0),
                            color=color, alpha=0.15, lw=0)
        ax.set_title(titles[metric], fontsize=10)
        ax.set_xlabel("α (0 = clean scores only, 1 = occluded only)")
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("verification accuracy (%)")
    axes[-1].legend(title="sparsity", fontsize=8, title_fontsize=8, frameon=False, loc="center left",
                    bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f"{method.upper().replace('_', '-')} one-shot α sweep (mean ± sd over scoring seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "alpha_sweep.png", dpi=200)
    plt.close(fig)
    return {"table": out / "alpha_sweep.csv", "markdown": out / "alpha_sweep.md", "figure": out / "alpha_sweep.png"}
