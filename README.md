# ECG-Biometrics-Bench  
### A Unified Framework for Realistic and Reproducible ECG Biometrics Evaluation

[![tests](https://github.com/MParvan/ecg-biometrics-bench/actions/workflows/tests.yml/badge.svg)](https://github.com/MParvan/ecg-biometrics-bench/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🚨 The Problem

Most ECG biometric systems report near-perfect performance (≈99% accuracy).

However, these results are often **misleading**.

A large portion of the literature relies on:
- Random intra-session train/test splits  
- Overlapping temporal segments  
- Closed-set evaluation protocols  

These practices introduce **data leakage** and artificially inflate performance.

➡️ When evaluated under realistic conditions (cross-session, unseen subjects), performance **drops dramatically**.

---

## 💡 Key Insight

### Performance Inflation Under Random Intra-Session Splitting

Randomly dividing temporally adjacent ECG samples from the same recording
session between training and testing can produce optimistic performance
estimates. The model may exploit session-specific or near-neighbor
similarities that are unlikely to remain stable across time or unseen
subjects.

In this project, **Random Split Fallacy** is used as shorthand for this
specific form of performance inflation. The term does not imply that every
random split is invalid. Its suitability depends on the intended experimental
claim and whether temporal or subject-level generalization is being assessed.

Across multiple datasets and models, we show:

- Intra-session (random split): ~95–100% accuracy  
- Cross-session / realistic: **significant degradation (often <70%)**

This suggests current methods may **not generalize to real-world deployment**.

---

## 🧠 What This Repository Provides

ECG-Biometrics-Bench is a **modular, reproducible benchmarking framework** for ECG-based biometric systems.

It standardizes the full pipeline:

- Dataset ingestion across heterogeneous ECG datasets  
- Signal preprocessing and segmentation  
- Model training (CNN, LSTM, hybrid architectures)  
- Evaluation under realistic biometric protocols  

The framework enables **fair comparison across models, datasets, and evaluation settings**.

---

## ⚙️ Key Features

- 🔄 **Unified Dataset Interface**
  - Supports multiple public ECG datasets (ECG-ID, PTB, CYBHi, MIT-BIH, etc.)
  - Handles heterogeneous formats and structures

- 🧩 **Modular Pipeline**
  - Filtering, segmentation, augmentation, modeling, evaluation
  - Easily extensible for new methods

- 📊 **Rigorous Evaluation Protocols**
  - Known-subject and subject-disjoint evaluation
  - Intra-session and cross-session evaluation
  - Long-term temporal evaluation

- 🧪 **Reproducible Experiments**
  - Config-driven pipeline
  - Consistent preprocessing and evaluation across datasets

- 📉 **Realistic Benchmarking**
  - Explicitly avoids data leakage
  - Highlights performance degradation under real conditions

---

## Evaluation Terminology

In this repository, **subject-disjoint evaluation** means that the identities
used for biometric evaluation are excluded from the identities used to train
the feature extractor.

This setting evaluates generalization to previously unseen subjects, but it
should not be confused with a complete open-set identification system.
Open-set identification additionally requires the system to detect or reject
a probe that does not correspond to any enrolled identity.

Accordingly:

- Tasks 1, 2, 5, and 6 evaluate subjects represented during model training.
- Tasks 3, 4, 7, and 8 evaluate subjects excluded from model training.
- Identification remains a 1:N search among the identities enrolled in the
  evaluation gallery.
- Verification evaluates genuine and impostor comparisons for the relevant
  evaluation cohort.

### Verification pair generation

Four policies are available. Requested budgets are exact decision counts, not
approximate targets.

| Mode | Decisions evaluated |
|---|---|
| `all` | Every genuine and every impostor comparison. |
| `all_genuine` | Every genuine comparison, plus impostors exhaustively up to `max_impostor_pairs`; beyond that cap, impostors are sampled uniformly without replacement. |
| `balanced` | Exactly `pair_sampling_budget` decisions, half genuine and half impostor, drawing with replacement within a class only when that class cannot supply its share. |
| `random` | Exactly `pair_sampling_budget` decisions drawn from the complete decision universe, with replacement only when the budget exceeds the universe size. |

The bundled paper-reproduction and model-comparison verification configs use
the same settings:

```yaml
pair_sampling_mode: all_genuine
max_impostor_pairs: 1000000
pair_sampling_seed: 42
```

### Verification operating points

TAR at 0.1% FAR is the mandatory headline operating point, so the framework
always injects FAR `0.001` into the resolved FAR set. `target_fars` requests
*additional* operating points; the shipped verification configurations request
`0.1`, `0.01`, and `0.0001`, giving the resolved set:

```
0.0001   0.001   0.01   0.1
```

Operating points are read from observed empirical thresholds. A requested FAR
is never produced by interpolation: when the number of impostor comparisons
cannot resolve it — the smallest non-zero empirical FAR is larger than the
target — the corresponding TAR is reported as unavailable rather than
estimated. The mandatory point is therefore always requested, but it can still
be unavailable on a small impostor sample.

### Identification ties

Identification ranks are pessimistic and exact. For probe `i` with genuine
identity `c*`:

```
rank_i = count( score_ic >= score_ic* )
```

Every gallery identity whose score equals or exceeds the genuine score counts
against the genuine identity, so a tie can never be resolved in the system's
favour. Comparison is on exactly represented values — no tolerance, no
`isclose`, no rounding — and Rank-1, Rank-5, and the CMC curve are all derived
from this one rank vector.

---

## 👥 Enrollment Templates

For any task that enrolls identities into a gallery (`use_template=True`),
`--enrollment_template_mode` selects how the enrolled observations become a
matchable representation:

- **`fusion`** (default): every enrollment observation for an identity is
  aggregated into a single template, using `--template_fusion_method` (mean,
  median, trimmed mean, a representative-beat selection, or no aggregation at
  all). This is the path every shipped configuration uses, and it is
  unaffected by the option below.
- **`multi_template`**: instead of aggregating, the framework stores a fixed
  number of representative enrollment observations per identity —
  `--num_templates_per_identity` — and keeps them as separate rows in the
  gallery.

`--template_size` keeps its existing meaning under both modes: the number of
enrollment observations available per identity (`None` uses all of them).
`num_templates_per_identity` is a separate, smaller number — how many of
those observations are actually kept as templates — and every enrolled
identity must have at least that many available observations, or the run
fails with the identities and counts that fell short rather than silently
enrolling with fewer.

Representatives are chosen with `farthest_first_cosine`, a deterministic
selection that has no random seed and never re-runs differently between
repetitions: the first representative is the enrollment observation closest
to the identity's overall (unit-normalized) direction, and each following
representative is whichever remaining observation is least similar, by
cosine distance, to the representatives already chosen. This spreads the
stored templates across the identity's variability instead of clustering
them around one typical beat.

At match time, a probe is scored against every stored template and the
identity's score is the maximum over its templates
(`--template_score_aggregation`, currently only `max`). When probe-side
fusion is also enabled (`--probe_fusion_size > 1`), fusion is applied first,
independently for each stored template, and only the resulting per-template
scores are reduced to one identity score by the maximum — never the other
order. Concretely, for probe beats `b` and an identity's templates `T_j`:

```
identity_score = max_j( mean_b( score(probe_b, T_j) ) )
```

Multi-template enrollment currently does not support
`--use_deployment_evaluation`: identity-level aggregation over several
templates changes the score distribution a deployment threshold would be
calibrated on, so the two options are mutually exclusive for now.

### Enrollment budget, probe fusion, and beat merging

Three settings are easy to confuse because all three combine several
observations. They act at different stages and are independent:

| Setting | Stage | Effect |
|---|---|---|
| `template_size` | Enrollment | How many enrollment observations per identity are used to build the template. |
| `probe_fusion_size` | Scoring | How many probe observations are fused into one decision. |
| `num_beats_to_merge` | Preprocessing | How many raw beats form a single input sample. |

`template_size` selects the enrollment observations *before* fusion, using a
deterministic ordering rather than a random draw. The selection also applies
when `template_fusion_method` is `none`, so a no-fusion gallery is limited to
the requested depth instead of silently enrolling every available observation.

`probe_fusion_size` controls score-side fusion; the shipped configurations use
`probe_fusion_size: 1`, i.e. one probe observation per decision. Larger values
form complete, non-overlapping groups from observations that share a subject,
session, record, and source segment, taken in source order; a trailing group
that cannot be filled is dropped rather than scored at a smaller depth.
Identification fuses the score vectors before ranking, and verification
averages the compatible group and template scores before the decision metric.

---

## 🗄️ Supported Datasets

The framework provides plug-and-play automated download and parsing for the following datasets:

1. **ECG-ID**: 90 subjects, 310 records, Lead I at 500 Hz. Each record stores the raw signal alongside a hardware-filtered copy; the framework reads the raw channel by default so that ECG-ID passes through the same preprocessing as the other datasets, whose recordings are unfiltered. Pass `--signal_type filtered` to read the stored filtered channel instead, keeping in mind that its filter is not documented and is applied in series with the pipeline's own band-pass. Only 20 of the 90 subjects were recorded on more than one day, so the long-term regimes evaluate 20 subjects rather than 90, with a median gap of 8 to 11 days between the last enrolment recording and the first probe and a maximum of 21 days. The six-month figure in the database documentation describes the collection period, not the interval any protocol measures.
2. **PTB**: 290 subjects (includes healthy controls and clinical pathologies).
3. **MIT-BIH Arrhythmia**: 47 subjects, continuous 30-minute Holter recordings.
4. **NSRDB**: 18 subjects, one continuous recording each of 23.1 to 26.0 hours at 128 Hz, the lowest rate here. Channels are named only `ECG1` and `ECG2` with no documented electrode placement; the framework reads `ECG1`, which carries the larger signal in 17 of the 18 records. Because the recordings differ in length by nearly three hours, a minute window is only usable if it ends before the shortest recording at 1,388 minutes — a later window silently drops the subjects whose recording stops sooner.
5. **PTB-XL**: Massive 21k+ clinical record database (10-second segments).
6. **HeartPrint**: 199 subjects, 1,539 records, single lead from the thumbs of both hands at 250 Hz, collected over ten years. Four session folders: `session1` and `session2` are separate visits, while `session3r` and `session3l` are the reading condition and the long interval of the *same* third visit. Because those two share 135 recordings, the loader refuses a protocol that enrols or trains on one and probes the other. Two subjects who attended only once have that single sitting filed under both `session1` and `session2`; the loader keeps the `session1` copy and drops the repeat, so `session1` to `session2` evaluates 197 subjects rather than 199.
7. **CYBHi**: Two collections published together, at 1000 Hz from dry off-the-person contact. The recordings carry substantial 50 Hz mains interference, so the pipeline band-pass is load-bearing rather than redundant. The short-term collection records 65 participants in one sitting, simultaneously on two units: `8B` from the hand palms with Ag/AgCl electrodes and `85` from the index and middle fingers with electrolycra, where mains hum dominates the spectrum. The loader reads one unit at a time — Ag/AgCl by default, since the electrolycra recordings mostly defeat beat detection — and `electrode_unit: "both"` pools them for an electrode-material comparison. Its three moments are one sitting: a briefing with no stimulus, then a low-arousal and a high-arousal video, so pairing them measures tolerance to induced arousal rather than to a session gap. The long-term collection records 63 subjects at the fingers in two visits roughly three months apart. A test acquisition named `VIDEOPRINT` that appears alongside the participants is excluded.

---

## 🗺️ The Regime Mapping Protocol

Public ECG datasets have incompatible temporal structures, so a single split
rule cannot apply to all of them. The regime mapping protocol decouples the
physical organization of a dataset from its logical biometric use. Datasets
fall into three categories, each with a different enrollment/probe rule.

### Category 1 — Record-order datasets (ECG-ID, PTB, PTB-XL)

Multiple recordings per subject with no uniform session structure. Recordings
are sorted by `(date, record order)` and partitioned deterministically. The
rules live in `select_record_order_partition` in [load_dataset.py](load_dataset.py)
and are used by both the loaders and the audit tool.

| `--data_split_mode` | Enrollment | Probe | Measures |
|---|---|---|---|
| `single-cross-session` | recording 1 | recording 2 | cross-record stability |
| `single-shot-short-term` | first recording of day 1 | remaining day-1 recordings | minimal enrollment, same day |
| `leave-last-out-short-term` | all day-1 recordings except the last | last day-1 recording | multi-shot enrollment, same day |
| `single-shot-long-term` | all day-1 recordings | all recordings from later days | template aging |
| `leave-last-out-long-term` | all recordings before the last day | all recordings on the last day | multi-session fusion vs aging |

Subjects lacking the structure a regime requires (for example, only one
recording when two are needed) are dropped, and the count is printed. In all
five regimes the enrollment partition consists only of recordings that precede
the probe recordings.

> **Note on PTB dates.** PTB records without a parseable timestamp receive
> synthetic sequential dates from a counter that advances across the dataset.
> Record *order* is therefore always well defined, but for those records the
> "long-term" regimes separate by file order rather than by measured elapsed
> time. ECG-ID and PTB-XL use real acquisition dates.

### Category 2 — Session-structured datasets (CYBHi, Heartprint)

Sessions are provided by the dataset, so they are used directly. Enrollment and
probe sessions are named explicitly and validated to be disjoint:

```bash
python main.py --dataset heartprint --task 6 \
  --data_split_mode cross-session \
  --train_sessions session1 session2 \
  --probe_sessions session3r
```

CYBHi exposes `short-term_CI`, `short-term_A1`, `short-term_A2`,
`long-term_S1`, and `long-term_S2`. Heartprint exposes `session1`, `session2`,
`session3l`, and `session3r`.

### Category 3 — Continuous recordings (MIT-BIH, NSRDB)

One continuous signal per subject, so file-based splitting does not apply.
Session-based evaluation is replaced by deterministic temporal windowing over
explicit minute ranges, validated to be non-overlapping:

```bash
python main.py --dataset mitbih --task 6 \
  --data_split_mode custom-split \
  --train_parts 0 5 --train_parts 12.5 17.5 \
  --enrol_parts 0 5 --enrol_parts 12.5 17.5 \
  --test_parts 25 30
```

The windows used for the reported results are listed in
[configs/paper_reproduction/README.md](configs/paper_reproduction/README.md).

---

## 🚀 Installation & Quick Start

To prevent dependency conflicts with other projects on your machine, it is highly recommended to install the framework inside a virtual environment.

**Option A: Using Python's built-in `venv`**
```bash
# 1. Create the virtual environment named "ecg_env"
python -m venv ecg_env

# 2. Activate the environment
# On Windows:
ecg_env\Scripts\activate
# On macOS and Linux:
source ecg_env/bin/activate
```
**Option B: Using Conda (Anaconda/Miniconda)**
```bash
# 1. Create the conda environment (Python 3.9+ recommended)
conda create -n ecg_env python=3.10 -y

# 2. Activate the environment
conda activate ecg_env
```
Once your virtual environment is activated, you can proceed with the standard installation:
```bash
git clone https://github.com/MParvan/ecg-biometrics-bench.git
cd ecg-biometrics-bench
pip install -r requirements.txt
```

Dependency versions are pinned to the environment used to produce the reported
results, running Python 3.10.

**CPU and GPU.** The same installation runs on either. The framework detects
the available device at runtime and falls back to CPU, and `--device cpu`
forces it. For GPU acceleration, install the CUDA build of PyTorch *before*
the requirements file, because the default PyPI wheel is CPU-only on Windows:

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Match the index URL to your CUDA version; see
[pytorch.org](https://pytorch.org/get-started/locally/). On CPU the framework
is fully functional but slow, so prefer a single dataset or a reduced epoch
count when exploring:

```bash
python -m scripts.reproduce_tables --dataset ecgid --run --smoke
```

**Optional representations.** The two-dimensional time–frequency
representations in `representation.py` (Mel-spectrogram, Gramian angular
fields, S-transform, Wigner–Ville) need extra packages that the core pipeline
never imports. They are kept separate because one of them builds a C
extension, and a failed build would otherwise abort the whole installation:

```bash
pip install -r requirements-representations.txt
```

**Running the tests.** The test suite needs one extra package:

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

The framework includes main.py which can simply be used as CLI. Some examples are as follows:

Experiment 1: Baseline Closed-Set Identification (Task 1)
Train a model to recognize known subjects from the ECG-ID dataset using the standard Softmax classifier.
```bash
python main.py \
  --dataset ecgid \
  --task 1 \
  --data_split_mode single-shot-short-term \
  --model deepecg \
  --epochs 150 \
  --batch_size 256 \
  --save_results
```

Experiment 2: Subject-Disjoint Verification with Template Matching (Task 4)
Train the feature extractor on Subject Group A, then evaluate 1:1 verification
on entirely unseen Subject Group B from the PTB dataset.
```bash
python main.py \
  --dataset ptb \
  --task 4 \
  --data_split_mode all-available \
  --use_template \
  --template_size 5 \
  --matching_method cosine \
  --pair_sampling_mode all_genuine \
  --max_impostor_pairs 1000000 \
  --pair_sampling_seed 42 \
  --save_results
```

Experiment 3: Cross-Session Temporal Robustness (Task 5)
Test how well the model handles physiological aging. Train/enroll subjects using their session1 recordings from the HeartPrint dataset, and probe them using their session2 recordings.
```bash
python main.py \
  --dataset heartprint \
  --task 5 \
  --data_split_mode cross-session \
  --train_sessions session1 \
  --probe_sessions session2 \
  --use_template \
  --template_fusion_method mean \
  --save_results
```

Experiment 4: The Ultimate Test - Disjoint Cross-Session Verification with SQI (Task 8)
The hardest biometric scenario. The model learns representations on Session 1 (Group A). It then enrolls unseen subjects (Group B) using their Session 1 data, and verifies them using their Session 2 data. We also enable dynamic Kurtosis SQI filtering to automatically drop noisy beats.
```bash
python main.py \
  --dataset cybhi \
  --task 8 \
  --data_split_mode cross-session \
  --train_sessions short-term_CI \
  --probe_sessions short-term_A2 \
  --use_template \
  --template_size 5 \
  --outlier_filtering_on_train \
  --outlier_filtering_on_test \
  --sqi_method kurtosis \
  --sqi_threshold 0.05 \
  --save_results \
  --visualize
```

You can simply run these experiments in the main folder of the project by opening a command prompt or powershell (if you are using a windows) and run the commands. Alternatively you can see the example usage of Experiment 1 using google colab: 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MParvan/ecg-biometrics-bench/blob/main/experiments/Experiment_1.ipynb)

This interactive notebook will automatically:
1. Clone the repository into the Colab environment.
2. Install the necessary dependencies from `requirements.txt`.
3. Execute the **Baseline Closed-Set Identification (Task 1)** command step-by-step so you can see the pipeline in action without setting up a local environment.


---

## 🏗️ Framework Architecture

![ECG-biometrics-bench architecture](docs/architecture.svg)

The framework decomposes the biometric pipeline into six stages with
standardized interfaces between them, so a change to one stage cannot
silently alter another. The vector source is [docs/architecture.svg](docs/architecture.svg)
and an editable Mermaid version is [docs/architecture.mmd](docs/architecture.mmd).

| Stage | Module | Responsibility |
|---|---|---|
| 1. Configuration | `main.py`, `config.yaml`, `configs/` | CLI and YAML resolution, argument validation, leakage guards |
| 2. Ingestion | `load_dataset.py`, `preprocessing.py`, `filtering.py` | Unified loader API for 7 datasets, regime mapping, filtering and segmentation |
| 3. Partitioning | `run.py`, `data_augmentation.py` | Subject-cohort and sample partitioning, training-only augmentation |
| 4. Learning | `models.py`, `run.py` | Architecture-agnostic training across 8 evaluation regimes |
| 5. Evaluation | `utils.py` | Template fusion, matching, identification and verification metrics |
| 6. Reporting | `run.py`, `scripts/` | Structured records, tables, figures, audits, artifact manifests |

---

## ✅ Recommended Minimum Reporting Protocol

Reported ECG biometric performance is not comparable across studies unless the
evaluation conditions are stated. We recommend that future work report the
following as a minimum. Every item is produced automatically by this framework.

**1. Subject-disjoint results, not only closed-set results.**
Closed-set accuracy measures whether a model can separate identities it was
trained on. It does not measure whether the learned representation
generalizes. Report both, and label them explicitly.

**2. Cross-session results wherever the dataset supports them.**
Same-session evaluation is a weak upper bound. If a dataset has session
structure, report the cross-session number as the headline result. If it does
not, say so rather than presenting a same-session number as if it were
comparable.

**3. Verification metrics, not identification accuracy alone.**
Rank-1 accuracy depends on gallery size and is not comparable across datasets
or cohort sizes. Report EER and TAR at a stated FAR alongside it. State the
FAR explicitly: `TAR@0.1%FAR`, never `TAR@FAR`.

**4. The exact dataset version and subject count used.**
Include the number of subjects actually evaluated after any filtering, and
state what was filtered and why. A result on a healthy-only subset is not
comparable to one on the full cohort.

**5. The complete preprocessing configuration.**
Filter type and cutoffs, R-peak detector, segmentation window, and
normalization. This framework stores the effective configuration in every
experiment record, so it can be quoted rather than reconstructed.

**6. Variability across seeds, with an interval.**
A single run is not a result. Report the mean over several seeds together with
a confidence interval, and use a paired test when comparing conditions that
share seeds.

**7. How enrollment and probe data were separated.**
State which recordings, sessions, or time windows formed the enrollment set
and which formed the probe set, and confirm they are disjoint. This is the
single most common source of inflated results.

**8. Released code and configurations.**
A configuration file that runs end to end is worth more than a methods
paragraph.

### Verifying items 6-8 in this framework

```bash
# 6. Confidence intervals are written into every experiment record
#    under "across_seed_uncertainty", and paired tests are available:
python -m scripts.statistical_comparisons --manifest comparison_manifest.yaml

# 7. Confirm enrollment never uses probe or future data, without training:
python -m scripts.audit_temporal_causality --config <configuration.yaml>

# 8. Reproduce a reported table from the shipped configurations:
python -m scripts.reproduce_tables --table 5 --run
python -m scripts.reproduce_tables --table 5 --collect
```

---

## 🎯 Beat Alignment

Beat-mode segmentation cuts a fixed window around each detected R-peak, so the
window is only correct if the detector reports the fiducial point itself.
Several widely used QRS detectors report the maximum of a smoothed detection
curve instead, which sits later than the true peak by roughly half the
smoothing window. For the Pan-Tompkins and Hamilton implementations reachable
through NeuroKit2 that is 40–70 ms. The offset cancels in interval
measurements, so it does not affect heart-rate or HRV analysis; it matters when
a fixed window is cut around the point.

`cut_beats` therefore refines each detection to the largest deflection nearby
before cutting. `align_window_s` sets how far it looks:

```yaml
preprocessing_parameters:
  align_peak: true
  align_window_s: 0.10   # half-width of the search, in seconds
```

Two properties are worth knowing:

- **The search runs on the recording, not inside the beat.** `pre_s` and
  `post_s` describe the segment that is cut once the peak is located, so they
  place no limit on how far the search may look. Only the ends of the recording
  do. The order is: detect, refine, then cut around the refined position.
- **Polarity is decided once per recording.** Taking the largest absolute
  deviation per beat handles inverted leads, but lets a deep S-wave outrank a
  modest R-peak on an individual beat. Comparing upward against downward
  deflection across the whole recording lets the many unambiguous beats settle
  the question for the ambiguous ones.

Measured against the beat annotations shipped with ECG-ID, MIT-BIH and NSRDB —
the three supported datasets that carry them, at 500, 360 and 128 Hz — a
half-width of 0.10 s brings every detector reachable through NeuroKit2 to
within one to three samples of the annotated peak. The width is a duration
rather than a fraction of the R-R interval, because the lag is a property of
each detector's smoothing window, which is defined in seconds.

Detector choice still governs how many beats are found and how many spurious
detections appear. Refinement substantially reduces how much it governs where
the beats are cut, on the datasets where that can be measured against
annotations.

---

## 🔒 Leakage Controls

The framework refuses configurations that would place the same physical
samples on both sides of an enrollment/probe comparison.

**Continuous recordings (MIT-BIH, NSRDB).** These are one continuous signal
per subject rather than discrete sessions, so partitions are explicit minute
windows. The loader validates that the enrollment coverage
(`train_parts` ∪ `enrol_parts`) does not intersect the probe coverage
(`test_parts`), and reports the realized windows and the achieved separation.
`train_parts` and `enrol_parts` are frequently identical by design, because the
framework uses the training partition as the gallery partition; overlap between
those two is reported but is not leakage.

An optional guard band additionally rejects windows that are merely adjacent:

```bash
python main.py --config <configuration.yaml> --temporal_guard_minutes 5
```

**Session-structured datasets (CYBHi, Heartprint).** Cross-session tasks
reject any configuration whose probe sessions also supply training or
enrollment data. Ordering is deliberately *not* enforced: a reverse protocol
that enrols on a later session and probes an earlier one measures directional
drift and is a legitimate experiment.

**Record-order datasets (ECG-ID, PTB, PTB-XL).** Regime selection is defined
once in `select_record_order_partition` and reused by every loader and by the
audit tool, so the audit reports the partitions that were actually evaluated
rather than a restatement of the intended protocol.

**Near-neighbour samples.** Merging `--num_beats_to_merge > 1` slides the merge
window one beat at a time by default, so neighbouring samples share beats. For
Tasks 1–4 the evaluation partitions are drawn from within one session, where
two overlapping samples could land on opposite sides of the boundary and share
raw beats across nominally independent roles. Those tasks therefore reject a
stride narrower than the merge width, and ask for non-overlapping samples
instead:

```bash
python main.py --dataset ecgid --task 1 --num_beats_to_merge 3 --beat_merge_stride 3
```

Tasks 5–8 separate their roles by session or recording rather than within a
session, so overlapping merge windows cannot cross a role boundary there and
the restriction does not apply. Independently of the task, a stride wider than
the merge width is rejected for every task, because it would silently discard
the beats between consecutive samples.

**Data roles and signal-quality filtering.** TRAIN, ENROLLMENT, and PROBE are
distinct configurable roles; the framework does not assume that training and
enrollment use the same data. Signal-quality filtering follows those roles:

- `outlier_filtering_on_train` acts on the representation-learning samples.
- When enrollment reuses the training role, the gallery naturally inherits
  those already-processed samples.
- When enrollment is supplied as its own role, it is independent of the
  train-side switch and reaches the gallery as provided.
- `outlier_filtering_on_test` controls probe-side filtering separately, and
  applies no per-subject ranking so that probe selection stays
  identity-independent.

There is deliberately no enrollment-side filtering switch: an explicitly
supplied gallery is the one the protocol asked for.

---

## 🔁 Reproducing the Reported Results

The `configs/paper_reproduction/` directory contains **150 self-contained YAML
configurations**, one per reported experiment. See
[its README](configs/paper_reproduction/README.md) for the shared settings and
the protocol definitions.

```bash
# Inspect the plan for one table without running anything
python -m scripts.reproduce_tables --table 5 --dry-run

# Verify the wiring end to end on CPU in minutes
python -m scripts.reproduce_tables --dataset ecgid --run --smoke

# Run one dataset for real
python -m scripts.reproduce_tables --dataset ecgid --run

# Assemble the tables from whatever has finished
python -m scripts.reproduce_tables --collect --output-dir reproduced_tables
```

Rows that have not been run yet are reported explicitly rather than silently
omitted.

### The experiment campaign

`campaigns/final_campaign.yaml` describes the complete experiment collection in
one file. The core campaign contains the complete shipped configuration corpus
and the additional single-factor study conditions derived from it. Study arms
reuse a shipped configuration's result as their baseline rather than
recomputing it.

```bash
# Check the manifest without running anything
python -m scripts.run_campaign validate --manifest campaigns/final_campaign.yaml

# Conditions and run executions, derived from the manifest
python -m scripts.run_campaign count --manifest campaigns/final_campaign.yaml

# The complete core campaign, including the shipped corpus and study additions
python -m scripts.run_campaign list --manifest campaigns/final_campaign.yaml --tier core

# Execute the complete core campaign
python -m scripts.run_campaign run --manifest campaigns/final_campaign.yaml \
    --tier core --artifact-root ../ecg-biometrics-artifacts

# Execute an optional study (optional studies must be named directly)
python -m scripts.run_campaign run --manifest campaigns/final_campaign.yaml \
    --study median_template_fusion --artifact-root ../ecg-biometrics-artifacts
```

`validate`, `count` and `list` never train, never read a dataset and never write
results. Execution requires an explicit `--tier` or `--study`; `run --tier
optional` is refused because every optional study must be named directly.
Listing the optional tier remains supported. With `--resume`, a condition is
skipped only when exactly one complete, publication-eligible result matches both
its scientific configuration and the current implementation provenance. An
output file on its own is never treated as completion.

### Seeds

Three seeds control independent sources of randomness.

| Setting | Controls |
|---|---|
| `seed` | Run and training stochasticity: weight initialization, batch shuffling, augmentation, and other randomness not governed by the other two. |
| `split_seed` | Randomized allocation of samples to data roles. |
| `pair_sampling_seed` | Verification pair sampling, where the configured mode samples rather than enumerates. |

With `n_runs > 1` the run seed advances by one per replicate, so the standard
five-run schedule starting at 42 is:

```
42   43   44   45   46
```

`split_seed` selects between two partition policies:

- **omitted or `null`** — the resolved split seed follows the run seed, so each
  replicate re-draws its partition and the reported spread includes allocation
  variability;
- **an explicit integer** — the partition is held fixed across replicates, so
  only training stochasticity varies.

Both are legitimate; they answer different questions, and the shipped
configurations leave `split_seed` unset deliberately.

`pair_sampling_seed` is separate from both. It does not advance with the run
index and does not affect partitioning, so verification comparisons can be
reproduced independently of how a run was trained or split.

### Result provenance

Every structured result record carries enough provenance to audit or reproduce
the numbers it contains, including the effective scientific configuration
identity, the implementation and source identity, the run seeds and the
resolved split seeds, data and run provenance, and references to trained
weights where a run produced them.

The publication-oriented consumers — the table, figure, manifest, and
statistical-comparison scripts — read that provenance and refuse artifacts that
do not meet the repository's publication-eligibility requirements, rather than
quietly reporting a number whose origin cannot be established. Exploratory work
can opt out: the table, figure, and manifest scripts accept
`--allow-exploratory-results`, and the statistical-comparison script reads
`allow_exploratory_results` from its manifest.

### Which artifacts exist

```bash
python -m scripts.build_artifact_manifest \
    --cache-dir ../ecg-biometrics-artifacts/cache \
    --output-markdown ARTIFACTS.md
```

This names every trained weight file with its dataset, protocol, seed,
epochs actually trained, SHA-256 checksum, and the table row it supports, plus
which of the 150 configurations have been executed.

### What drives temporal drift

```bash
python -m scripts.analyze_drift_covariates \
    --dataset ecgid --data_split_mode leave-last-out-long-term \
    --output-csv drift_ecgid.csv --output-json drift_ecgid.json
```

For each subject this relates cross-session degradation to the covariates that
are genuinely measurable: elapsed time between enrollment and probe, change in
mean heart rate from the detected RR intervals, and the amplitude ratio between
the two templates. It reports univariate associations, a joint standardized
model, and variance inflation factors.

By default the outcome is **model-free**: one minus the correlation between a
subject's enrollment and probe beat templates. That measures drift in the
signal itself rather than in one architecture's embedding, so the result is not
specific to any model. Supply `--subject-scores` with a per-subject EER file to
explain a model-based outcome instead.

**Read the limitations the report prints.** Of the factors usually named as
causes of ECG template aging, only some are identifiable from public data:

| Factor | Status |
|---|---|
| Elapsed time | Measurable in every integrated dataset |
| Heart-rate change | Measurable in every integrated dataset |
| Health status | Available only for PTB and PTB-XL; the other cohorts are healthy by construction and the covariate has no variance |
| **Electrode shift** | **Not annotated anywhere.** Amplitude change responds to it but also to physiological state and recording gain, so it cannot isolate placement |

More fundamentally, these factors change **together** between sessions and no
public dataset varies them independently. The analysis therefore reports
associations, never causal contributions, and the variance inflation factors
show directly when two covariates are too entangled for either coefficient to
be read alone. Isolating the drivers would require a dataset that deliberately
varies electrode placement, elapsed time, and physiological state
independently. No such ECG biometric dataset currently exists.

### Publication figures

```bash
python -m scripts.make_figures --dataset ecgid --metric EER --figure degradation
python -m scripts.make_figures --dataset ecgid --metric EER --figure comparison
python -m scripts.make_figures --dataset heartprint --metric EER --figure paired \
    --left short_term --right long_term
```

Figures are written as PDF and PNG (add `--format svg` for SVG), use a palette
validated for colour-vision deficiency with hatching as a secondary encoding,
carry 95% confidence intervals rather than standard deviations, and annotate
paired comparisons with the statistics described below. Use `--font-size` to
scale all type at once.

### Paired comparisons

The figure and statistical-comparison scripts share one implementation, so a
figure and the table beside it cannot disagree.

Repeated runs are paired **by run seed**, never by row order: the two
conditions must cover exactly the same seeds, and a missing counterpart is an
error rather than a silently dropped observation. For each comparison the
scripts report a paired t-test (`scipy.stats.ttest_rel`), a Wilcoxon
signed-rank test under a pinned deterministic policy, and Cohen's *d_z*
computed from the paired differences with the sample standard deviation
(`ddof=1`).

Multiple comparisons are corrected with the Holm step-down procedure. The
parametric and non-parametric tests are corrected as **separate families**, so
a Wilcoxon result can never influence a t-test decision. A family is fixed by
the analysis rather than by whichever rows a run happens to produce: one
comparison manifest defines one family per test type across the hypotheses it
declares, and one paired-figure invocation defines one family per test type
across its evaluation-setting panels.

Significance markers on figures use the **Holm-adjusted** paired-t p-value, not
the raw one, so a marker reflects the same evidence as the reported decision.
Raw and adjusted p-values both remain in the statistics outputs.

With five runs the Wilcoxon test cannot reach p < 0.05 in a two-sided test
regardless of effect size, so it is reported as a distribution-free companion
to the t-test rather than as the deciding statistic.

---

## 🧪 Tests

The framework ships a comprehensive pytest suite covering the parts where a
silent error would corrupt a result rather than raise: enrollment/probe
separation, protocol and configuration invariants, cache and provenance
identity, metric arithmetic, and the analysis utilities.

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

No test reads the ECG datasets. Loaders are either mocked or given a synthetic
WFDB fixture, so the suite runs on a clean checkout in seconds without the
multi-gigabyte download.

Some tests exist specifically to stop a class of mistake recurring:

| Test module | Guards against |
|---|---|
| `test_continuous_temporal_partitions.py` | Enrollment and probe windows overlapping in a continuous recording |
| `test_temporal_causality_audit.py` | A template being built from data at or after the probe |
| `test_cache_identity_stability.py` | A new option silently invalidating every cached array and weight file |
| `test_across_seed_uncertainty.py` | Aggregate metrics being stored as unparseable text |
| `test_literature_baselines.py` | A model that cannot accept every dataset's beat length |
| `test_requirements.py` | An import that is not declared, or a dependency that is not pinned |
| `test_tutorial_notebooks.py` | Tutorials drifting out of step with the API |

## 🐳 Container

A CPU image is defined in [Dockerfile](Dockerfile), for running the benchmark
without installing anything locally.

```bash
docker build -t ecg-biometrics-bench .
docker run --rm ecg-biometrics-bench -m pytest tests -q
```

Mount the dataset and artifact directories so downloads and results persist
between runs:

```bash
docker run --rm \
    -v "$(pwd)/datasets:/app/datasets" \
    -v "$(pwd)/artifacts:/artifacts" \
    ecg-biometrics-bench \
    main.py --config configs/paper_reproduction/ecgid/ecgid_all_available_closed_set_task01_identification.yaml
```

The image installs the CPU build of PyTorch. For GPU execution, run on the
host using the CUDA build described above, or derive an image from an
`nvidia/cuda` base.

## ⚙️ Continuous integration

[`.github/workflows/tests.yml`](.github/workflows/tests.yml) runs on every push
and pull request:

- the test suite on Python 3.10 and 3.12
- validation that all 234 shipped configurations still parse
- a container build, with the suite executed inside the resulting image

---

## Tutorials

See [experiments/README.md](experiments/README.md) for the full index.

* **Dataset loading**: loading and using each of the seven datasets. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MParvan/ecg-biometrics-bench/blob/main/experiments/load_dataset_Module.ipynb)
* **Protocol switching**: driving the eight evaluation regimes through `run.py`. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MParvan/ecg-biometrics-bench/blob/main/experiments/run_Module.ipynb)
* **Custom model integration**: the contract a model must satisfy, and the one-line registration. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MParvan/ecg-biometrics-bench/blob/main/experiments/Custom_Model.ipynb)
* **Custom dataset**: benchmarking your own data. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MParvan/ecg-biometrics-bench/blob/main/experiments/Custom_Dataset.ipynb)

### Adding a model

Registration is one dictionary entry in `MODEL_REGISTRY` in [main.py](main.py).
The `--model` choices are generated from its keys, so a new tag becomes
available to all eight protocols and appears in `--help` automatically. Your
model needs `__init__(in_channels, num_classes, include_top)`, must return
logits when `include_top=True` and a 2-D embedding when `False`, and must use
adaptive pooling so it accepts the 76–600 sample range the datasets produce.

```bash
# Check a custom model against the same contract tests the built-ins pass
python -m pytest tests/test_literature_baselines.py -k ModelContract
```

### Benchmarking several architectures

`configs/model_comparison/` holds 84 configurations that evaluate seven
architectures under identical protocols, including re-implementations of
published ECG biometric methods. See
[its README](configs/model_comparison/README.md).

---

## Citation

If you use ECG-Biometrics-Bench in your research, please cite:

```bibtex
@article{parvan2026ecg,
  title   = {ECG-biometrics-bench: A Unified Framework for Reproducible
             Benchmarking of ECG Biometrics},
  author  = {Parvan, Milad},
  journal = {arXiv preprint arXiv:2605.01548},
  year    = {2026}
}
```


