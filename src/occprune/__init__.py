"""Occlusion-aware gradient pruning for robust face verification (CS 260D).

Module map (dependencies point downwards only):

    Experiments (entry points called by notebooks / CLI)
        pipeline              data preparation, Dense-Clean / Dense-Aug training and evaluation   (notebooks 01, 02)
        prune_run             one-shot pruning + optional fine-tuning, all seven methods          (notebook 03)
        calibration_ablation  calibration-photo selection study                                  (notebook 04)
        prune_report          tables, paired bootstrap, clean-vs-occluded importance analysis

    Building blocks
        training              memmap batches, optimizer groups, lr schedule, CosFace step, model init
        protocol              split / checkpoint / occlusion integrity checks, calibration draw, counts
        pruning               prunable layers, Random / Magnitude / Taylor scores, OAGP aggregation, masks
        calibration_selection facility location, k-center, GreeDi, Greedy-Merge, Sieve-Streaming selectors
        faceverif             embedding net, CosFace head, pair occlusions, embeddings, 10-fold verification
        data                  CASIA / LFW caches, held-out identity split, validation pairs
        occlusion, transforms training-time square occlusion, uint8 -> normalised tensors
        utils                 config, seeds, devices, timers, JSON, environment info
"""

__version__ = "0.1.0"
