# Architecture comparison configurations

84 configurations that benchmark **seven architectures** under identical
protocols, so that published ECG biometric methods can be compared on equal
terms rather than against a single baseline.

## What varies and what does not

Each file is a clone of a row from `configs/paper_reproduction/`, with only
two fields changed: `model` and `results_dir`. Preprocessing, seeds, epochs,
template fusion, matching, and pair sampling are identical across all seven
architectures, so any difference in the result is attributable to the
architecture.

## Architectures

| `--model` | Source | Notes |
|---|---|---|
| `deepecg` | Labati et al., *Pattern Recognition Letters* 126 (2019) 78-85 | Default architecture |
| `ecgxtractor` | Melzi et al., *IEEE Access* 11 (2023) 15555-15566 | Autoencoder-style encoder; the framework's closest prior comparator |
| `mobilenet_gru` | Rai & Kafley, arXiv:2509.20382 (2025) | Depthwise-separable trunk plus bidirectional GRU |
| `multiscale_cnn` | Chu et al., *IEEE Access* 7 (2019) 51598-51607 | Parallel branches at three receptive fields |
| `separable_resnet` | Ihsanto et al., *Applied Sciences* 10 (9) (2020) 3304 | Residual depthwise-separable CNN |
| `resnet1d` | — | Generic 1D ResNet |
| `hybrid` | — | CNN-LSTM |

These are **re-implementations written from the architecture descriptions in
the cited papers**, not the authors' released code. Two adaptations are applied
uniformly so that all seven can be evaluated under one protocol:

1. Global average pooling replaces any flatten-then-dense stage, because beat
   length ranges from 76 samples (NSRDB) to 600 (PTB) and a fixed flatten
   would tie each model to one dataset.
2. The classifier head is separable, so one trained network supplies both
   closed-set logits and verification embeddings.

Consequently these numbers characterise each architecture **under this
benchmark's protocol**. They are not reproductions of the originally published
results, which used different preprocessing, different splits, and in most
cases different datasets. Any comparison against a published figure should
state that difference.

## Protocols covered

The long-term cross-session protocols for ECG-ID, Heartprint, and CYBHi, in
all four task variants (closed-set and subject-disjoint × identification and
verification). These are the conditions where architectural differences are
most likely to show, because same-session protocols saturate.

## Running

```bash
# Inspect the plan
python -m scripts.reproduce_tables --config-root configs/model_comparison --dry-run

# Verify the wiring on CPU first
python -m scripts.reproduce_tables --config-root configs/model_comparison --run --smoke

# Run one architecture
python main.py --config configs/model_comparison/ecgid/ecgid_ecgxtractor_leave_last_out_long_term_closed_set_task06_verification.yaml
```

Because `model` is part of the weight-cache key, each architecture trains its
own feature extractor and none can reuse another's weights.

## Reporting

`scripts/make_figures.py` groups by protocol and setting rather than by model,
so for an architecture comparison, collect the records and tabulate them:

```bash
python -m scripts.reproduce_tables --config-root configs/model_comparison \
    --collect --output-dir model_comparison_tables
```

To test whether one architecture is genuinely better than another rather than
luckier, use the paired comparison, which aligns the two configurations by
seed:

```bash
python -m scripts.statistical_comparisons --manifest <manifest.yaml>
```
