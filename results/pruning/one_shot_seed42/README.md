# Dense-Aug one-shot pruning on LFW

Training seed 42, scoring seeds 42, 43 and 44. All methods start from the same [CASIA Dense-Aug checkpoint](https://github.com/yhr04/CS260D_Final_Project/releases/tag/dense-baselines-seed42). We compare four fixed region masks on the second face in each pair. The primary contrast, OAGP-max minus α-weighted OAGP on the four-region mean, averaged **+1.23 percentage points** (sample SD 0.33 pp). The corresponding contrast against magnitude was **-1.15 pp**.

| Method | Clean (%) | Mean of square 10% and 30% | Square 50% | Mean of four regions |
| --- | ---: | ---: | ---: | ---: |
| Dense-Aug | 98.75 | 97.61 | 90.70 | 95.77 |
| Magnitude | 98.57 | 97.21 | 89.17 | 94.93 |
| OAGP (α=0.5) | 97.89 | 95.73 | 85.55 | 92.55 |
| OAGP-max | 97.96 | 96.26 | 89.37 | 93.78 |

Each gradient-method row averages the three scoring seeds. Dense and magnitude use the common deterministic control from seed 42. The region mean gives equal weight to eyes, mouth/nose, upper half and lower half. Each region uses all 6,000 LFW pairs and the team's 10-fold verification protocol. Square 50% participated in CASIA validation checkpoint selection.

| Scoring seed | OAGP region mean (%) | OAGP-max region mean (%) | Max minus OAGP (pp) |
| ---: | ---: | ---: | ---: |
| 42 | 92.18 | 93.76 | +1.583 |
| 43 | 92.68 | 93.85 | +1.171 |
| 44 | 92.80 | 93.73 | +0.925 |

The [independent stability check](stability.json) recomputed 960 fold accuracies from the saved pair scores.

## Reproduction and scope

- Use notebook 03 or the README command with the release weights, 50% per-layer sparsity, B=1,024, α=0.5, zero fine-tuning steps and evaluation batch size 64. Keep the split, source order, pair masks and settings fixed across methods.
- For each scoring seed, score 512 clean and 512 square-occluded CASIA training photos. Compute the mean absolute per-example weight-times-gradient of the CosFace loss. Normalize within each condition and layer. OAGP takes a weighted mean; OAGP-max selects the larger score per weight. The two methods share all 1,024 backward passes. Median shared scoring time was 14.5 seconds on the local Apple MPS device.
- All pruned models retain 5,578,752 of 11,157,504 eligible convolution weights. The stem, embedding and classifier head keep their weights. Stage timings and counts are in [counts_and_timing.csv](counts_and_timing.csv).
- We ran fresh inference with flipped embeddings and bf16 on Apple MPS. Measured dense clean accuracy was 98.75% (published 98.72%). The largest absolute difference across the eight LFW conditions was 0.25 pp. Only the scoring sample varies between the three repeats. The experiment uses one trained model, one sparsity and synthetic occlusion.
- [metrics.csv](metrics.csv) holds per-condition LFW and held-out CASIA accuracy, fold SD, AUC and TAR at FAR 0.001, as fractions. [mask_overlap.csv](mask_overlap.csv) gives the per-layer intersection and union of the kept OAGP and OAGP-max coordinates. [pair_similarities.npz](pair_similarities.npz) contains cosine scores and the shared labels and folds. The score key format and file checksums are in [protocol.json](protocol.json).
