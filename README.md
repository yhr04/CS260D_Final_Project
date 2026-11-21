# Occlusion-Aware Gradient Pruning for Robust Face Identification

UCLA CS 260D (Large-Scale Machine Learning), Fall 2026 — final project.

We ask whether **parameter importance estimated on both clean and synthetically occluded faces** preserves
occluded-face accuracy better than conventional criteria (magnitude, clean-gradient Taylor) when a
ResNet-18 is pruned to the same sparsity with the same gradient-scoring budget.

Project plan, experimental design and timeline: [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) (Chinese).

> This is **closed-set identification** on LFW-funneled identities with ≥ 20 images. Its accuracy is
> not the official LFW face-verification benchmark.

## Status

Two-tier design (details in `docs/PROJECT_PLAN.md`):
**main** benchmark = train on CASIA-WebFace (112×112), test on LFW face verification with clean and occluded pairs
(the protocol used in the occluded-face-recognition literature, e.g. PDSN, FROM);
**dev** benchmark = closed-set identification on 62 LFW identities, for fast development and multi-seed runs.

| Tier | Item | Status |
|---|---|---|
| dev | Data, split, fixed occlusion masks | done |
| dev | Dense-Clean / Dense-Aug baselines | done (seed 42, 15 epochs; rerun with 25 epochs planned) |
| dev | Pruning (magnitude, Clean/Occ/Mixed-Taylor, OAGP, OAGP-max) | next |
| main | CASIA-WebFace download, memmap cache, held-out validation pairs, fixed occlusions (M0) | done |
| main | CosFace Dense-Clean / Dense-Aug, LFW/AgeDB/CALFW/CPLFW clean + occluded verification (M1) | see `results/casia/` |
| main | Pruning comparison, region occlusions, ablations | planned |

## Dev-tier dense baseline results (seed 42, test set, %)

| Method | Clean | Occ 10% | Occ 30% | Occ 50% |
|---|---|---|---|---|
| Dense-Clean | 92.68 | 62.60 | 24.61 | 18.11 |
| Dense-Aug | 93.13 | 92.68 | 85.51 | 64.52 |

RTX 3070 Ti Laptop GPU, ~21 s of training per model. Balanced accuracy, robustness drop and diagnostics:
`results/dense/*.json`, `results/baseline_table_seed42.csv`, `figures/`.

## Main-tier dense baseline results (CASIA-WebFace -> LFW verification, seed 42, %)

LFW 6000 pairs, standard 10-fold accuracy (mean ± std); the 2nd image of every pair is occluded.
Squares = same distribution as training occlusion; regions = never seen in training.

| Method | Best ep. | Clean | Sq 10% | Sq 30% | Sq 50% | Eyes | Mouth+nose | Upper half | Lower half |
|---|---|---|---|---|---|---|---|---|---|
| Dense-Clean | 18 | 98.88 ± 0.41 | 97.08 ± 0.56 | 83.73 ± 1.04 | 65.62 ± 1.61 | 92.77 ± 1.41 | 90.10 ± 1.84 | 92.00 ± 1.42 | 86.30 ± 2.80 |
| Dense-Aug | 20 | 98.72 ± 0.44 | 98.77 ± 0.46 | 96.52 ± 0.54 | 90.65 ± 1.03 | 96.93 ± 0.41 | 96.02 ± 1.21 | 94.87 ± 0.89 | 94.83 ± 0.95 |

ResNet-18 + CosFace, 20 epochs, ~42 / 54 min of training on an RTX 3070 Ti Laptop GPU. AgeDB-30 / CALFW / CPLFW,
paired bootstrap CIs, TAR@FAR and gradient diagnostics: `results/casia/`, figures `figures/casia_dense_*`.

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

Check: `python -m pytest -q` (16 tests) and
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
| `notebooks/04_casia_data.ipynb` | main tier: download instructions, cache build, held-out split, fixed pair occlusions, throughput (`prepare_casia`) |
| `notebooks/05_casia_dense.ipynb` | main tier: CosFace Dense-Clean / Dense-Aug on CASIA-WebFace (`run_face_dense`, resumable), LFW verification table, bootstrap |

Main-tier data (~4.1 GB, no Kaggle login needed), kept zipped in `data/_downloads/`:
```bash
curl -L -o data/_downloads/webface.zip https://www.kaggle.com/api/v1/datasets/download/yakhyokhuja/webface-112x112
curl -L -o data/_downloads/val.zip     https://www.kaggle.com/api/v1/datasets/download/yakhyokhuja/agedb-30-calfw-cplfw-lfw-aligned-112x112
```
Long main-tier runs can also be started from a terminal: `python -m occprune.casia_pipeline train --mode aug`
(with `src` on `PYTHONPATH`); notebook and CLI call the same functions and share the resume checkpoint.

All logic lives in `src/occprune/pipeline.py`. In notebook 02, `RETRAIN = True` retrains (otherwise saved
results are reused). Hyper-parameters come from `configs/default.yaml` and can be overridden in the settings
cell (`cfg["train"]["epochs"] = 25`, ...). Other seeds: set `SEED = 43` (split, masks and training seed change together).

## Layout

```
configs/default.yaml      every hyper-parameter of the dev tier
configs/casia.yaml        every hyper-parameter of the main tier
src/occprune/
  data.py                 LFW loading (sklearn), per-identity 70/15/15 split, split persistence
  occlusion.py            square-occlusion boxes: training sampler, fixed eval masks (NumPy only)
  transforms.py           uint8 -> [0,1] -> occlusion -> flip -> resize -> ImageNet normalisation
  models.py               ResNet-18 (ImageNet weights), checkpoints, parameter counts
  train.py                shared training loop (also used for post-pruning fine-tuning)
  evaluate.py             accuracy / balanced accuracy per occlusion level, train-set gradient diagnostics
  pipeline.py             experiment steps called by the notebooks (dev tier)
  casia.py                main tier: zip -> uint8 memmap caches, held-out identity split + validation pairs
  faceverif.py            main tier: ResNet-18 embedding net, CosFace head, pair occlusions, 10-fold verification
  casia_pipeline.py       main tier: prepare / throughput / resumable training / evaluation / summary (+ CLI)
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
