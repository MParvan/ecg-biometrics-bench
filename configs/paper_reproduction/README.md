# Paper reproduction configurations

**150 self-contained YAML configurations**, one per reported experiment row, covering ECG-ID, PTB, PTB-XL, MIT-BIH, NSRDB, CYBHi, and Heartprint. Together they reproduce every published result table.

Each YAML is directly executable:

```bash
python main.py --config configs/paper_reproduction/ecgid/ecgid_all_available_closed_set_task01_identification.yaml
```

To run a whole table, or to assemble a table from results that already
exist, use the driver:

```bash
python -m scripts.reproduce_tables --table 5 --dry-run
```

## Naming

Filenames encode dataset, protocol, evaluation setting, task number, and task type. The setting reported as `Open set` in the paper appears here as `subject_disjoint`, corresponding to Tasks 3, 4, 7, and 8.

## Shared settings

- DeepECG, 250 epochs, batch size 256, Adam learning rate 0.001
- Five runs with base seed 42, producing seeds 42-46
- No validation split and no augmentation
- Beat segmentation from 0.2 s before to 0.4 s after each R-peak
- Pan-Tompkins R-peak detection
- Butterworth band-pass filtering from 0.5 to 40 Hz, order 4
- Z-score normalization
- Mean template fusion using all available enrollment beats (`template_size: null`)
- Probe fusion size 3 and cosine matching. Verification uses
  `pair_sampling_mode: all_genuine`: every genuine comparison is retained,
  while at most 1,000,000 impostor comparisons are sampled uniformly without
  replacement (`max_impostor_pairs: 1000000`, `pair_sampling_seed: 42`).
- ECG-ID filtered hardware channel (`signal_type` appears only in the ECG-ID
  files, because no other loader reads it)
- Data cache and weight cache both enabled
- One isolated result directory per YAML

### A note on the weight cache

`intelligent_weight_loading: true` matches the configuration used to produce
the reported numbers. It lets the identification and verification variants of
a protocol share one trained feature extractor instead of training it twice,
which roughly halves the compute for a full reproduction.

It cannot change a result. The weight cache key includes the loader identity,
the training hyperparameters, the seed, and a SHA-256 fingerprint of the
post-partition, post-augmentation training arrays, so a cache hit can only
occur for a run that would have trained on exactly the same data with exactly
the same settings. To force training from scratch, override it:

```bash
python main.py --config <configuration.yaml> --no_intelligent_weight_loading
```

Every boolean option a configuration switches on has a matching `--no_<option>`
flag, so any YAML setting can be overridden from the command line without
editing the file.

Cache entries are content-addressed, so a stale entry cannot be silently
reused after a configuration change — the key simply stops matching.

### Leakage controls

Every configuration here passes the framework's enrollment/probe leakage
checks:

- The MIT-BIH and NSRDB minute windows are validated to be disjoint between
  the enrollment coverage (`train_parts` plus `enrol_parts`) and the probe
  coverage (`test_parts`). `train_parts` and `enrol_parts` are deliberately
  identical because the framework uses the training partition as the gallery
  partition.
- The CYBHi and Heartprint cross-session configurations draw probes only from
  sessions that supply neither training nor enrollment data.
  `heartprint_short_term_reverse_*` deliberately enrols on session 2 and
  probes session 1, measuring drift in the reverse temporal direction. The
  partitions remain disjoint, so this is a valid protocol rather than
  leakage.
- The record-order regimes for ECG-ID, PTB, and PTB-XL build templates only
  from recordings that precede the probe recordings.

The last two claims can be verified without training anything:

```bash
python -m scripts.audit_temporal_causality --config configs/paper_reproduction/ecgid/ecgid_leave_last_out_long_term_closed_set_task05_identification.yaml
python -m scripts.audit_temporal_causality --config configs/paper_reproduction/mitbih/mitbih_multi_shot_closed_set_task05_identification.yaml
```

## Multi-session enrollment semantics

The framework uses the training partition as the enrollment/gallery partition, so multi-shot configurations pool all historical enrollment windows into `train_parts` or `train_sessions`:

- MIT-BIH multi-shot: 0-5 and 12.5-17.5 minutes; probe 25-30 minutes.
- NSRDB multi-shot: 0-5, 120-125, 360-365, and 720-725 minutes; probe 1380-1385 minutes. The probe window ends before the shortest recording, which runs to 1388 minutes, so all 18 subjects are probed rather than only those whose recording extends further.
- Heartprint S1-S2-S3R and S1-S2-S3L: sessions 1 and 2 are pooled for training/enrollment; session 3R or 3L is the probe.

A protocol that instead keeps separate historical training and enrollment partitions will not produce identical numbers, because the gallery is then built from different data.

## File counts

- `ecgid/`: 28 YAML files
- `ptb/`: 28 YAML files
- `ptbxl/`: 14 YAML files
- `mitbih/`: 16 YAML files
- `nsrdb/`: 16 YAML files
- `cybhi/`: 20 YAML files
- `heartprint/`: 28 YAML files

PTB-XL identification configurations are intentionally omitted: dense pairwise score matrices over its 18,885-subject cohort are prohibitively memory-hungry, so identification metrics are not reported for that dataset.
