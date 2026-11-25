# Occlusion-Aware Gradient Pruning for Robust Face Verification

UCLA CS 260D (Large-Scale Machine Learning), Fall 2026 — final project.

We ask whether **parameter importance estimated on both clean and synthetically occluded faces** preserves
occluded-face verification accuracy better than conventional criteria (magnitude, clean-gradient Taylor) when a
ResNet-18 face model is pruned to the same sparsity with the same gradient-scoring budget.

## Results and notebooks

| Experiment | Read the results | Open the notebook |
| --- | --- | --- |
| 50% one-shot pruning | [Dense-Aug, Magnitude, OAGP and OAGP-max](results/pruning/one_shot_seed42/README.md) | [03 · Run any of the seven pruning methods](notebooks/03_pruning.ipynb) |
| Calibration photo selection for OAGP and OAGP-max | [Six selectors at 256, 512 and 1,024 photos](results/pruning/calibration_selection/README.md) | [04 · Explore results or rerun pruning](notebooks/04_calibration_selection.ipynb) |
| Random pruning at 50% | [Accuracy by occlusion across three seeds](results/random50/lfw_summary.csv) | [03 · Random only](notebooks/03_random_pruning.ipynb) |
| Dense baselines | [Dense-Clean and Dense-Aug verification results](results/baseline_table_seed42.md) | [02 · Train and evaluate](notebooks/02_dense_baselines.ipynb) |

Notebook 04 opens the saved results without a GPU, weights or face images. To run pruning from the
checkpoint, follow the [environment setup](#environment), [download the data](#data), build the cache
with [01 · Data preparation](notebooks/01_data.ipynb) and get the [Dense-Aug weights](#trained-models).
For implementation details, see the [pruning scores and masks](src/occprune/pruning.py),
[calibration selectors](src/occprune/calibration_selection.py) and [default settings](configs/default.yaml).

**Main finding.** With 50% layerwise sparsity and zero fine-tuning, Magnitude reached **94.93%** mean LFW
verification accuracy across four held-out facial occlusions. At 1,024 calibration photos, Sieve-Streaming
gave the highest mean among the six selectors for OAGP (**92.97%**) and OAGP-max (**93.87%**).
On the 50%-area square alone, OAGP-max with Sieve reached **89.51%** and Magnitude reached **89.17%**.

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
| E2–E4 | Pruning: all 7 methods in one runner (`prune_run`), sparsity sweep, main comparison, importance analysis | runner done; first 50% one-shot comparison done |
| E5+ | Region occlusions, α / budget ablations, optional extensions | calibration selection and budget comparison complete at 50% sparsity; α sweep planned |

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
python -m occprune.prune_run --checkpoint checkpoints/dense_aug_seed42.pt --output results/pruning/runs/e3_score42     --sparsity 0.7 0.8 --scoring-seed 42 --ft-epochs 1 --test-set lfw
```

Defaults (B=1024, α=0.5, fine-tuning lr 0.01 with cosine decay) live in the `prune` section of
`configs/default.yaml`. Leave out `--test-set lfw` while choosing settings on the held-out CASIA pairs.
Each output directory holds `result.json` and `pair_similarities.npz` (tracked) plus `condition_scores.npz`
and `masks.npz` (large, git-ignored). `occprune.prune_report` builds the tables (`metrics_table`,
`summary_table`), paired bootstrap CIs and the clean-vs-occluded importance analysis (`bank_agreement`).

All logic lives in `src/occprune/`; notebooks only call it. In notebook 02, `RETRAIN = True` retrains (otherwise
saved results are reused). Hyper-parameters come from `configs/default.yaml` and can be overridden in the
settings cell. Long runs can also be started from a terminal (same functions, same resume checkpoint):
```bash
python -m occprune.pipeline train --mode aug      # with src on PYTHONPATH
```

## Layout

```
configs/default.yaml      every hyper-parameter in one place
src/occprune/
  data.py                 zip -> uint8 memmap caches, held-out identity split + validation pairs, test-pair caches
  faceverif.py            ResNet-18 embedding net, CosFace head, pair occlusions, embeddings, 10-fold verification
  pipeline.py             prepare / throughput / resumable training / evaluation / summary (+ CLI)
  occlusion.py            training-time square occlusion sampler (NumPy only)
  transforms.py           uint8 -> [0,1] -> occlusion -> flip -> ImageNet normalisation
  utils.py                seeds, device, synchronised timers, config, JSON, env info
notebooks/                entry point (01 data, 02 dense baselines, 03 pruning, ...)
tests/                    pytest quality checks
results/                  JSON/CSV results, per-pair similarities, split, fixed occlusions, logs   (tracked)
figures/                  PNG/PDF figures                                                          (tracked)
checkpoints/, data/       model weights (see "Trained models"), downloads and caches               (git-ignored)
requirements.txt          pinned versions for pip / conda
pyproject.toml, uv.lock   same dependencies for uv
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
