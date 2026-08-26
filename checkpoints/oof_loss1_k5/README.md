# Loss×1 Group 5-Fold checkpoints

- Model: ViT-B/16 + CLAdapter, mean pooling
- Anomaly Loss multiplier: 1.0
- Checkpoint selection: inner Validation anomaly-class AP
- Ensemble: uniform mean of five anomaly probabilities
- Frozen OOF threshold: `0.013258215`

The exact SHA256 values and reference 230-image diagnostic result are recorded
in `operating_rule.json`.
