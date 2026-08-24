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
- Pan-Tompkins R-peak detection, each detection refined to the largest
  deflection within `align_window_s: 0.10` seconds before the beat is cut. The
  search runs on the recording, so `pre_s` and `post_s` do not limit it.
- Butterworth band-pass filtering from 0.5 to 40 Hz, order 4
- Z-score normalization
- Mean template fusion. Tasks 5 to 8 draw enrollment and probe from separately
  defined partitions, so `template_size: null` there means every beat in the
  enrollment partition. Tasks 3 and 4 evaluate subjects the representation
  never saw, and both the gallery and the probes come from those same subjects,
  so the budget spent on the gallery decides how many beats are left to probe
  with. Those files state `template_size: 1`, and leaving it unset is rejected.
- Probe fusion size 1 and cosine matching: one probe observation is
  evaluated per biometric decision. Enrollment-template fusion is a separate
  concept covered by `template_size` and `template_fusion_method` above.
  Multi-beat probe fusion is available as an optional analysis for both
  identification and template-based verification: setting `probe_fusion_size`
  above one averages that many probe-score observations from consecutive
  beats within one source block (subject, session, record, segment) into a
  single fused decision. The framework currently implements score-level
  arithmetic-mean probe fusion. Verification uses `pair_sampling_mode:
  all_genuine`: every genuine comparison is retained, while at most 1,000,000
  impostor comparisons are sampled uniformly without replacement
  (`max_impostor_pairs: 1000000`, `pair_sampling_seed: 42`).
- ECG-ID raw channel (`signal_type: raw`). The database stores a raw and a
  hardware-filtered channel; the raw one is read so that the only filtering
  applied is the band-pass above, rather than that band-pass in series with an
  undocumented hardware filter.
- CYBHi short-term acquiring unit `8B`, the Ag/AgCl palm electrodes
  (`electrode_unit: "8B"`). Every short-term acquisition was recorded by two
  units at once, so pooling them would mix two electrode configurations into a
  single identity. The long-term collection was acquired by a single unit and
  takes no such setting.
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
- Those regimes read meaning into the order of the acquisition dates, so a
  recording whose header states no date takes no part in them. Such recordings
  are not given a stand-in date, which would let an unknown acquisition time
  stand as evidence of elapsed time; they remain available to `all-available`,
  which draws on every recording without ordering them. In PTB this affects 8
  of 549 records, 7 of which belong to patients holding no other recording. The
  eighth belongs to patient180, who keeps six dated recordings across four
  distinct days and so stays eligible throughout. The long-term cohort is 92
  subjects either way. Each loader reports the count at load time.

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

PTB-XL identification configurations are intentionally omitted: dense pairwise score matrices over its 18,869-patient cohort are prohibitively memory-hungry, so identification metrics are not reported for that dataset.
