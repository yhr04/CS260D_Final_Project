# Occlusion-Aware Gradient Pruning for Robust Face Verification

UCLA CS 260D (Large-Scale Machine Learning), Fall 2026 — final project.

We ask whether **parameter importance estimated on both clean and synthetically occluded faces** preserves
occluded-face verification accuracy better than conventional criteria (magnitude, clean-gradient Taylor) when a
ResNet-18 face model is pruned to the same sparsity with the same gradient-scoring budget.

## Results and notebooks

| Experiment | Read the results | Open the notebook |
| --- | --- | --- |
| One-shot pruning, 50–90% sparsity, 7 methods × 3 seeds | [Sparsity curves, paired contrasts, clean vs occluded importance](results/pruning/oneshot_grid/README.md) | [03 · Pruning runner and saved results](notebooks/03_pruning.ipynb) |
| Pruning + 1 epoch of fine-tuning at 60% and 70% | [Recovery after fine-tuning](results/pruning/finetune_grid/README.md) | [03 · Pruning runner and saved results](notebooks/03_pruning.ipynb) |
| OAGP fusion weight α | [α ∈ {0, 0.25, 0.5, 0.75, 1}, chosen on validation](results/pruning/alpha_sweep/README.md) | [03 · Pruning runner and saved results](notebooks/03_pruning.ipynb) |
| Calibration photo selection for OAGP and OAGP-max | [Six selectors at 256, 512 and 1,024 photos](results/pruning/calibration_selection/README.md) | [04 · Explore results or rerun pruning](notebooks/04_calibration_selection.ipynb) |
| Dense baselines | [Dense-Clean and Dense-Aug verification results](results/baseline_table_seed42.md) | [02 · Train and evaluate](notebooks/02_dense_baselines.ipynb) |

Earlier 50% runs, kept for reference and superseded by the grids above:
[one-shot Dense-Aug / Magnitude / OAGP / OAGP-max on Apple MPS](results/pruning/one_shot_seed42/README.md) and
[Random at 50%](results/random50/lfw_summary.csv) ([notebook](notebooks/03_random_pruning.ipynb)).

Notebook 04 opens the saved results without a GPU, weights or face images. To run pruning from the
checkpoint, follow the [environment setup](#environment), [download the data](#data), build the cache
with [01 · Data preparation](notebooks/01_data.ipynb) and get the [Dense-Aug weights](#trained-models).
For implementation details, see the [pruning scores and masks](src/occprune/pruning.py),
[calibration selectors](src/occprune/calibration_selection.py) and [default settings](configs/default.yaml).

**Main findings** (LFW, mean accuracy over four held-out facial occlusions, three scoring seeds; Dense-Aug 95.66%):

1. **Occluded calibration data beats clean data for gradient-based importance.** Without fine-tuning, Occ-Taylor beat
   Clean-Taylor by +6.2 / +11.5 / +4.8 pp at 50 / 60 / 70% sparsity (pair-level CIs above zero in 22–24 of 24 cells).
   After one epoch of fine-tuning the gap shrinks to about +0.2 pp but never reverses.
2. **Magnitude is the most robust one-shot criterion.** It kept 93.21% at 60% and 86.68% at 70%, while every Taylor
   variant fell below 80% and 60%.
3. **Fine-tuning erases most differences.** At 60–70% sparsity all methods except Random recover to 95.0–95.4%.
4. **Fusing clean and occluded scores does not help.** OAGP-max beats α-weighted OAGP, validation picks α = 0.75–1
   (mostly occluded), and OAGP-max ≈ Occ-Taylor. Clean and occluded importance are closely aligned (Spearman 0.96;
   79–89% of kept weights shared).
5. **Choosing calibration photos matters little.** At 50% sparsity the best of six coreset-style selectors (Sieve-Streaming)
   gains only +0.41 / +0.09 pp over random photos for OAGP / OAGP-max, within seed variation.

**Protocol** (as in the occluded-face-recognition literature, e.g. PDSN, FROM): train on **CASIA-WebFace**
(112×112 aligned, 10.5K identities, 481K training images), test **face verification on LFW** (6000 pairs, standard
10 folds) under clean and occluded conditions. Only the second image of each pair receives an occlusion.
Test squares cover 10%, 30% or 50% of the image, with 10–30% used for training augmentation.
Four template-based regions are held out from training (eyes, mouth+nose, upper / lower half).
AgeDB-30, CALFW and CPLFW are supplementary test sets.

Project plan, experimental design and timeline: [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) (Chinese).

## Status

| ID | Item | Status |
|---|---|---|
| E0 | Data download, memmap cache, held-out validation pairs, fixed test occlusions, throughput | done |
| E1 | Dense-Clean / Dense-Aug (ResNet-18 + CosFace), clean + occluded verification | done (seed 42) |
| E2 | Sparsity sweep 50–90%, all 7 methods, no fine-tuning | done (3 scoring seeds) |
| E3 | Main comparison at 60% / 70%, one-shot and after 1 epoch of fine-tuning | done (3 scoring seeds) |
| E4 | Clean vs occluded importance: Spearman, kept-weight overlap per layer | done |
| E5 | Held-out region occlusions | done (in every table) |
| E6 | OAGP α sweep {0, 0.25, 0.5, 0.75, 1} | done; OAGP-SNR and estimator ablations not run |
| E7 | Three scoring seeds | done |
| E8 / E10 | Calibration budget 256 / 512 / 1,024 and six photo selectors | done at 50% sparsity |
| — | Not run: second dense training seed, fine-tuning seeds, AgeDB / CALFW / CPLFW for pruned models, E9 (Wanda), E11 (real occlusion) | listed as limitations |

## Dense baseline results (seed 42, LFW verification accuracy, %)

10-fold mean ± std; the 2nd image of every pair is occluded.

| Method | Best ep. | Clean | Sq 10% | Sq 30% | Sq 50% | Eyes | Mouth+nose | Upper half | Lower half |
|---|---|---|---|---|---|---|---|---|---|
| Dense-Clean | 18 | 98.88 ± 0.41 | 97.08 ± 0.56 | 83.73 ± 1.04 | 65.62 ± 1.61 | 92.77 ± 1.41 | 90.10 ± 1.84 | 92.00 ± 1.42 | 86.30 ± 2.80 |
| Dense-Aug | 20 | 98.72 ± 0.44 | 98.77 ± 0.46 | 96.52 ± 0.54 | 90.65 ± 1.03 | 96.93 ± 0.41 | 96.02 ± 1.21 | 94.87 ± 0.89 | 94.83 ± 0.95 |

ResNet-18 + CosFace, 20 epochs, 41 / 54 min of training on an RTX 3070 Ti Laptop GPU. AgeDB-30 / CALFW / CPLFW,
paired-bootstrap CIs, TAR@FAR=1e-3, AUC and gradient diagnostics: `results/dense/`, `results/baseline_table_seed42.md`,
figures `figures/dense_*`.

## Environment

Any Python 3.12 environment with the pinned libraries works (3.11–3.13 should too). Pick **one** option.

<details>
<summary>Installation commands for pip, conda or uv</summary>

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

</details>

Check: `python -m pytest -q` and
`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

<details>
<summary>Select the notebook kernel in VS Code</summary>

In VS Code open the **repository folder** itself, open a notebook, then
*Select Kernel → Python Environments* and pick the environment you just created. If it is not listed,
reload the window (`Ctrl+Shift+P` → *Developer: Reload Window*), or register it once as a named kernel
from inside that environment:
```bash
python -m ipykernel install --user --name cs260d --display-name "CS260D"
```
and choose *Select Kernel → Jupyter Kernel… → CS260D*. The first cell of each notebook prints the Python
path in use; it must be the environment you installed into. The notebooks add `src/` to `sys.path`
themselves, so the project package does not need to be installed.

</details>

## Data

~4.1 GB from Kaggle (no login needed), kept zipped in `data/_downloads/` (git-ignored):
```bash
curl -L -o data/_downloads/webface.zip https://www.kaggle.com/api/v1/datasets/download/yakhyokhuja/webface-112x112
curl -L -o data/_downloads/val.zip     https://www.kaggle.com/api/v1/datasets/download/yakhyokhuja/agedb-30-calfw-cplfw-lfw-aligned-112x112
```
Notebook 01 decodes them once into `data/cache/` (uint8 memmap, ~18.5 GB, ~1.5 min).

## Trained models

The dense checkpoints are not in git (`*.pt` is ignored). They are attached to the GitHub release
[`dense-baselines-seed42`](https://github.com/yhr04/CS260D_Final_Project/releases/tag/dense-baselines-seed42);
download them into `checkpoints/` instead of retraining:

| File | Size | Needed for | SHA-256 |
|---|---|---|---|
| `dense_aug_seed42.pt` | 67 MB | **every pruning experiment** (shared starting point) | `d8248b5dda150101f1b80188ad58d5b9b23977abd1843e58657f0cd5ef001bc3` |
| `dense_clean_seed42.pt` | 67 MB | only to re-evaluate the Dense-Clean baseline | `abb4f24ca85f5970035eaef1eea56049a21d75c3c3381d395d6e112764408066` |

```bash
curl -L -o checkpoints/dense_aug_seed42.pt   https://github.com/yhr04/CS260D_Final_Project/releases/download/dense-baselines-seed42/dense_aug_seed42.pt
curl -L -o checkpoints/dense_clean_seed42.pt https://github.com/yhr04/CS260D_Final_Project/releases/download/dense-baselines-seed42/dense_clean_seed42.pt
```
Check the hash with `certutil -hashfile checkpoints\dense_aug_seed42.pt SHA256` (Windows) or `sha256sum` (macOS/Linux).
Each file holds the embedding network (`model`), the CosFace head (`head`, needed for gradient-based scoring)
and the training config (`meta`); load it with
`occprune.pipeline.load_checkpoint("checkpoints/dense_aug_seed42.pt")`. Only load checkpoints from this
release: `.pt` files are unpickled. The data (previous section) is still needed for evaluation and pruning.

## Running experiments

Run the [notebooks](#results-and-notebooks) top to bottom. Explanations are in Chinese. Figures,
tables and logs are in English for the report.

### Pruning runs

The [pruning runner](src/occprune/prune_run.py) supports Random, Magnitude, Clean-Taylor, Occ-Taylor,
Mixed-Taylor, OAGP and OAGP-max. It computes each score once and reuses it at every sparsity.

```bash
python -m occprune.prune_run --checkpoint checkpoints/dense_aug_seed42.pt --output results/pruning/runs/finetune_grid_score42 \
    --sparsity 0.6 0.7 --scoring-seed 42 --ft-epochs 1 --test-set lfw
```

Each results README under `results/pruning/` lists the exact commands behind its runs. One-shot runs take about
10–20 minutes per seed on an RTX 3070 Ti Laptop GPU, and fine-tuning adds about 2 minutes per method and sparsity.

Defaults (B=1024, α=0.5, fine-tuning lr 0.01 with cosine decay) live in the `prune` section of
`configs/default.yaml`. Leave out `--test-set lfw` while choosing settings on the held-out CASIA pairs.
Each output directory holds `result.json` and `pair_similarities.npz` (tracked) plus `condition_scores.npz`
and `masks.npz` (large, git-ignored). `occprune.prune_report` builds the tables (`metrics_table`,
`summary_table`), paired bootstrap CIs and the clean-vs-occluded importance analysis (`bank_agreement`).
`write_grid_report` writes every table and figure for a set of runs, and `write_alpha_report` builds the α sweep.

All logic lives in `src/occprune/`; notebooks only call it. In notebook 02, `RETRAIN = True` retrains (otherwise
saved results are reused). Hyper-parameters come from `configs/default.yaml` and can be overridden in the
settings cell. Long runs can also be started from a terminal (same functions, same resume checkpoint):
```bash
python -m occprune.pipeline train --mode aug      # with src on PYTHONPATH
```

## Layout

```
configs/default.yaml        every hyper-parameter in one place (dense training + `prune` section)
src/occprune/               see the module map in src/occprune/__init__.py
  pipeline.py               data preparation, dense training / evaluation / summary (+ CLI)        notebooks 01, 02
  prune_run.py              pruning runner: seven methods, shared calibration, fine-tuning (+ CLI)  notebook 03
  calibration_ablation.py   calibration-photo selection study                                     notebook 04
  prune_report.py           pruning tables, paired bootstrap, importance agreement
  training.py               memmap batches, optimizer groups, lr schedule, CosFace step
  protocol.py               split / checkpoint / occlusion integrity checks, calibration draw, counts
  pruning.py                prunable layers, scores (Random, Magnitude, Taylor), OAGP aggregation, masks
  calibration_selection.py  coreset-style calibration selectors (+ _sieve_selection.py)
  faceverif.py              embedding net, CosFace head, pair occlusions, embeddings, 10-fold verification
  data.py                   zip -> uint8 memmap caches, held-out identity split + validation pairs
  occlusion.py, transforms.py, utils.py   occlusion sampler, preprocessing, config / seeds / timers / JSON
notebooks/                  entry points (01 data, 02 dense baselines, 03 pruning, 04 calibration selection)
scripts/                    standalone checks (verify_calibration_selection.py, NumPy only)
tests/                      pytest quality checks
results/                    JSON/CSV results, per-pair similarities, split, fixed occlusions, logs   (tracked)
figures/                    PNG/PDF figures                                                          (tracked)
checkpoints/, data/         model weights (see "Trained models"), downloads and caches               (git-ignored)
requirements.txt            pinned versions for pip / conda
pyproject.toml, uv.lock     same dependencies for uv
```

## Key protocol decisions

* **LFW is test-only.** 2% of CASIA identities (211 ids, 9,413 images) are held out of training; 3000 positive +
  3000 negative pairs on them select the epoch (and later α etc.). Split: `results/splits/split_seed42.npz`.
* **Occluded probe vs clean gallery.** Only the 2nd image of every pair is occluded. Random squares use fixed seeds;
  region occlusions use fixed coordinates of the 112×112 alignment template. All are saved in `results/masks/`
  and shared by every method.
* **Standard verification metric.** Cosine similarity of flip-augmented, L2-normalised embeddings; threshold chosen
  on 9 folds, accuracy on the 10th. Per-pair similarities are saved for paired bootstrap / McNemar tests.
* **Same recipe for both dense models.** Same initial weights (incl. the CosFace head), batch order, flips, optimiser
  and epochs; the only difference is training-time occlusion. Model selection = epoch with the best mean held-out
  accuracy over clean + 10/30/50% squares, for both models.
* **Native resolution, bf16.** 112×112 input, no resizing; bf16 autocast for the trunk, CosFace logits in fp32.
