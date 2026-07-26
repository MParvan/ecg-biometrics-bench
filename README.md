# ECG-Biometrics-Bench  
### A Unified Framework for Realistic and Reproducible ECG Biometrics Evaluation

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

---

## 🗄️ Supported Datasets

The framework provides plug-and-play automated download and parsing for the following datasets:

1. **ECG-ID**: 90 subjects, varied short-term/long-term sessions. The framework uses the hardware-filtered ECG channel by default; pass `--signal_type raw` to select the unfiltered channel explicitly.
2. **PTB**: 290 subjects (includes healthy controls and clinical pathologies).
3. **MIT-BIH Arrhythmia**: 47 subjects, continuous 30-minute Holter recordings.
4. **NSRDB**: 18 subjects, extremely long-term 24-hour continuous recordings.
5. **PTB-XL**: Massive 21k+ clinical record database (10-second segments).
6. **HeartPrint**: Highly structured multi-session dataset (Baseline, Post-Exercise, Cognitive tasks, Temporal separation).
7. **CYBHi**: Designed for short-term and long-term (3-month gap) biometric stability.

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
conda create -n ecg_env python=3.12 -y

# 2. Activate the environment
conda activate ecg_env
```
Once your virtual environment is activated, you can proceed with the standard installation:
```bash
git clone https://github.com/MParvan/ecg-biometrics-bench.git
cd ecg-biometrics-bench
pip install -r requirements.txt
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
  --num_pairs 10000 \
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

## Tutorials
* **Working with load_dataset Module**: The colab notebook shows how you can load and use different datasets for different scenarios. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MParvan/ecg-biometrics-bench/blob/main/experiments/load_dataset_Module.ipynb)
* **Working with run Module**: The colab notebook shows how you can work with run.py module to evaluate your method/model in various biometric protocols. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MParvan/ecg-biometrics-bench/blob/main/experiments/run_Module.ipynb)
* **Injecting Your Custom PyTorch Model**: The colab notebook shows how you can add your model to the framework. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MParvan/ecg-biometrics-bench/blob/main/experiments/Custom_Model.ipynb)
* **Benchmarking Your Own Custom Dataset**: The colab notebook shows how you can evaluate methods in the framework on your own dataset. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MParvan/ecg-biometrics-bench/blob/main/experiments/Custom_Dataset.ipynb)

---

## Citation

If you use ECG-Biometrics-Bench in your research, please cite our paper:



