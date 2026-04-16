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

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/MParvan/ecg-biometrics-bench.git
cd ecg-biometrics-bench
pip install -r requirements.txt
