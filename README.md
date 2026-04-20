# 🫀 ECG-Biometrics-Bench: A Unified Framework for Reproducible Benchmarking

**ECG-Biometrics-Bench** is an open-source, unified framework designed to standardize the evaluation of ECG biometric systems. It provides an end-to-end pipeline from raw signal ingestion to rigorous biometric evaluations, including subject-disjoint and cross-session scenarios.

---

## ✨ Core Contributions & Features

* **Standardized Evaluation Matrix**: Implements 8 biometric protocols that independently isolate intra-session, cross-session (temporal robustness), closed-set, and subject-disjoint scenarios.
* **Unified Dataset API**: Built-in, automated loaders for 7 major public ECG databases, handling complex chronological partitioning and cross-session routing natively. 
* **Baseline Architectures**: Includes configurable deep learning models ranging from standard 1D-CNNs to Transformers.

---

## 🗄️ Supported Datasets

The framework provides plug-and-play automated download and parsing for the following datasets:

1. **ECG-ID**: 90 subjects, varied short-term/long-term sessions.
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

Experiment 2: Open-Set Verification with Template Matching (Task 4)
Train on Subject Group A, then test 1:1 verification on entirely unseen Subject Group B from the PTB dataset. We will use the first 5 beats to form an enrollment template.
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



