# Occlusion-Aware Gradient Pruning for Robust Face Identification

UCLA CS 260D (Large-Scale Machine Learning), Fall 2026 — final project.

We ask whether **parameter importance estimated on both clean and synthetically occluded faces** preserves
occluded-face accuracy better than conventional criteria (magnitude, clean-gradient Taylor) when a
ResNet-18 is pruned to the same sparsity with the same gradient-scoring budget.

Project plan, experimental design and timeline: [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) (Chinese).

> This is **closed-set identification** on LFW-funneled identities with ≥ 20 images. Its accuracy is
> not the official LFW face-verification benchmark.

## Status

| Phase | Item | Status |
|---|---|---|
| 1 | Data, split, fixed occlusion masks | done |
| 1 | Dense-Clean / Dense-Aug baselines | done (seed 42) |
| 1 | Pruning (magnitude, Clean-Taylor, Occ-Taylor, OAGP) | next |
| 2 | Sparsity sweep, alpha ablation, seeds | planned |

## Dense baseline results (seed 42, test set, %)

| Method | Clean | Occ 10% | Occ 30% | Occ 50% |
|---|---|---|---|---|
| Dense-Clean | 92.68 | 62.60 | 24.61 | 18.11 |
| Dense-Aug | 93.13 | 92.68 | 85.51 | 64.52 |

RTX 3070 Ti Laptop GPU, ~21 s of training per model. Balanced accuracy, robustness drop and diagnostics:
`results/dense/*.json`, `results/baseline_table_seed42.csv`, `figures/`.

## Environment

Any Python 3.12 environment with the pinned libraries works (3.11–3.13 should too). Pick **one** option.

**pip + venv**
```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
# NVIDIA GPU (Windows/Linux) - CUDA build of PyTorch first:
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

**conda**
```bash
conda create -n cs260d python=3.12 -y
conda activate cs260d
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU only
pip install -r requirements.txt
```

**uv** (uses `pyproject.toml` + `uv.lock`, picks the CUDA build automatically on Windows/Linux)
```bash
uv sync
```

Check: `python -m pytest -q` (10 tests) and
`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

**Notebook kernel.** In VS Code open the **repository folder** itself, open a notebook, then
*Select Kernel → Python Environments* and pick the environment you just created. If it is not listed,
reload the window (`Ctrl+Shift+P` → *Developer: Reload Window*), or register it once as a named kernel
from inside that environment:
```bash
python -m ipykernel install --user --name cs260d --display-name "CS260D"
```
and choose *Select Kernel → Jupyter Kernel… → CS260D*. The first cell of each notebook prints the Python
path in use; it must be the environment you installed into. The notebooks add `src/` to `sys.path`
themselves, so the project package does not need to be installed.

## Running experiments

The notebooks are the entry point; run them top to bottom. Explanations inside are in Chinese; figures,
tables and logs are in English for the report.

| Notebook | What it does |
|---|---|
| `notebooks/01_data_and_occlusion.ipynb` | downloads LFW, creates the split and fixed masks (`prepare_data`), inspects data and occlusion |
| `notebooks/02_dense_baselines.ipynb` | trains Dense-Clean and Dense-Aug (`run_dense`), result table, figures, paired bootstrap, gradient diagnostics |

All logic lives in `src/occprune/pipeline.py`. In notebook 02, `RETRAIN = True` retrains (otherwise saved
results are reused). Hyper-parameters come from `configs/default.yaml` and can be overridden in the settings
cell (`cfg["train"]["epochs"] = 25`, ...). Other seeds: set `SEED = 43` (split, masks and training seed change together).

## Layout

```
configs/default.yaml      every hyper-parameter in one place
src/occprune/
  data.py                 LFW loading (sklearn), per-identity 70/15/15 split, split persistence
  occlusion.py            square-occlusion boxes: training sampler, fixed eval masks (NumPy only)
  transforms.py           uint8 -> [0,1] -> occlusion -> flip -> resize -> ImageNet normalisation
  models.py               ResNet-18 (ImageNet weights), checkpoints, parameter counts
  train.py                shared training loop (also used for post-pruning fine-tuning)
  evaluate.py             accuracy / balanced accuracy per occlusion level, train-set gradient diagnostics
  pipeline.py             experiment steps called by the notebooks
  utils.py                seeds, device, synchronised timers, config, JSON, env info
notebooks/                main entry point (01 data & occlusion, 02 dense baselines, ...)
tests/                    pytest quality checks
results/                  JSON/CSV results, split indices, fixed masks, logs   (tracked)
figures/                  PNG/PDF figures                                      (tracked)
checkpoints/, data/       model weights and the LFW download                   (git-ignored)
requirements.txt          pinned versions for pip / conda
pyproject.toml, uv.lock   same dependencies for uv
```

## Key protocol decisions

* **Split before occlusion.** Original images are split per identity (70/15/15, every identity in every subset);
  indices are saved in `results/splits/`.
* **Fixed evaluation masks.** Val/test boxes are generated once with independent seeds and saved in
  `results/masks/`; every model is evaluated on the same boxes (3 masks per test image per level, averaged).
  The std over masks is *mask* variability, not uncertainty over images — use the saved per-image predictions
  for paired bootstrap / McNemar tests.
* **Native resolution.** A 128×128 square crop of the funneled image is fed to the network at 128 px
  (sklearn's default crop is 125×94; upsampling to 224 adds compute, not information). Set
  `data.img_size: 224` to follow the original plan.
* **Same recipe for both dense models.** Same initial weights (incl. the new head), batch order, flips,
  optimiser and epochs; the only difference is training-time occlusion. Model selection = epoch with the best
  *mean validation accuracy over all occlusion levels*, for both models. The test set is never used for selection.
* **Deterministic.** Re-running with the same seed on the same machine reproduces every number exactly
  (verified: two runs gave bit-identical histories and test results).
