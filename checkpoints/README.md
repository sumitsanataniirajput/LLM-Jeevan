# Checkpoints

Training outputs belong here by convention and are excluded from Git. No checkpoint, pretrained weights, or real training data are included. The Phase 4 trainer creates checkpoints only when explicitly run with user-supplied, owned, public-domain, or properly licensed data.

The latest checkpoint is resumable and contains model/optimizer state, step, scheduler metadata, RNG state, configs, and tokenizer. The best validation checkpoint is written separately with `.best` before the extension. Metrics use a `.metrics.csv` suffix by default.
