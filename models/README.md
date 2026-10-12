# Public pretrained resources

This directory holds public feature extraction models, not the sentiment models trained for this project. Only README files belong in Git; downloaded weights and configuration files remain local.

| Directory | Resource | Used for |
| --- | --- | --- |
| [bert-base-uncased/](bert-base-uncased/README.md) | Frozen BERT encoder and tokenizer | Q1 text features and Q2/Q3 text representations |
| [wav2vec2-base-960h/](wav2vec2-base-960h/README.md) | English acoustic model | Q1 word alignment and optional Q3 text-time candidates |
| [mediapipe/](mediapipe/README.md) | Face Landmarker | Q1 facial features |

From the project root, download the pinned versions and verify their hashes:

```powershell
pixi run resources
```

Existing files with matching hashes are reused. Versions and SHA-256 hashes are recorded in [the resource manifest](../docs/预训练资源校验.json).

The project's three trained sentiment checkpoints are already included in [outputs/q2_prediction/q2_systematic/](../outputs/q2_prediction/q2_systematic/): `gate_recon_seed2026.pt`, `gate_recon_seed2027.pt`, and `gate_recon_seed2028.pt`. Keep them in that location; they total approximately 7.85 MB.
