<div align="center">
# 🔩 CWRU Multi-Label Dataset Splitter

**A data-leakage-free, multi-label dataset division for the CWRU Bearing Fault Dataset**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/arXiv-2407.14625-b31b1b?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2407.14625)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](https://claude.ai/chat/LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-CWRU-f59e0b?style=flat-square)](https://engineering.case.edu/bearingdatacenter)

<br/>
>
> *Based on:* **"Benchmarking deep learning models for bearing fault diagnosis using the CWRU dataset: A multi-label approach"**
> Rosa et al., 2024 — [arXiv:2407.14625](https://arxiv.org/abs/2407.14625)
>

</div>
---


--

--

--

---

## 📌 Overview

The **Case Western Reserve University (CWRU)** bearing fault dataset is one of the most widely used benchmarks in rotating machinery fault diagnosis. However, the  **standard dataset divisions found in the literature suffer from data leakage** , producing over-optimistic results that fail to generalize to real-world scenarios.

This repository implements the dataset division proposed by Rosa et al. (2024), which:

* ✅ **Eliminates bearing-level data leakage** — signals from the same physical bearing never appear in both train and test sets
* ✅ **Formulates the problem as multi-label classification** — detects the presence or absence of each fault type (Inner race, Outer race, Ball) at each location (Drive End, Fan End) independently
* ✅ **Places all healthy signals exclusively in the test set** — solving the healthy-class leakage that prior multi-class approaches cannot avoid
* ✅ **Maximizes training diversity** — random fault size selection per location–type pair ensures diverse conditions in every split

---

## 🧠 Why Multi-Label?

Traditional approaches frame bearing diagnosis as **multi-class classification** (one label per sample:  *Healthy / Inner / Outer / Ball* ). This has several drawbacks:

| Issue                                       | Multi-Class | Multi-Label |
| ------------------------------------------- | :---------: | :---------: |
| Allows simultaneous faults                  |     ❌     |     ✅     |
| Avoids healthy-class leakage                |     ❌     |     ✅     |
| Handles class imbalance                     |     ❌     |     ✅     |
| Uses prevalence-independent metrics (AUROC) |     ❌     |     ✅     |
| Requires synchronous signals                |     ❌     |     ✅     |

In the multi-label formulation, each signal is assigned **three binary labels** — one per fault type — at the location where it was acquired:

```
Label = [IR_present, OR_present, Ball_present]

Healthy  →  [0, 0, 0]
Inner    →  [1, 0, 0]
Outer    →  [0, 1, 0]
Ball     →  [0, 0, 1]
```

---

## 🗂️ Dataset Structure

This tool expects the CWRU dataset to be organized as follows:

```
CWRU/
├── 12k_Drive_End_Bearing_Fault_Data/
│   ├── IR/
│   │   ├── 007/
│   │   ├── 014/
│   │   └── 021/
│   ├── OR/
│   │   ├── 007/
│   │   ├── 014/
│   │   └── 021/
│   └── B/
│       ├── 007/
│       ├── 014/
│       └── 021/
├── 12k_Fan_End_Bearing_Fault_Data/
│   └── ... (same structure as Drive End)
└── Normal/
    └── *.mat
```

> **Configurations used:** Loads of 1–3 HP · Fault sizes 7, 14, 21 mils · 12 kHz sampling rate (healthy resampled from 48 kHz) · Outer race faults: "Centered @6:00" preferred, "Orthogonal @3:00" as fallback.

The output directory will be organized as:

```
multi_label_cwru/
├── IR/
│   ├── DE/
│   │   ├── train/
│   │   └── test/
│   └── FE/
│       ├── train/
│       └── test/
├── OR/  ...
├── B/   ...
└── Normal/
    └── test/
```

---

## ⚙️ Dataset Division Strategy

The key insight of this split is that  **signals from the same physical bearing must never be shared across train and test sets** .

For each of the 6 (location × fault-type) pairs:

```
(DE, IR) · (DE, OR) · (DE, B) · (FE, IR) · (FE, OR) · (FE, B)
```

One fault size (007 / 014 / 021) is **randomly selected** to go into the test set. All signals of the remaining sizes go into the training set. This yields `3⁶ = 729` possible splits.

All  **healthy bearing signals are always placed in the test set only** , making this formulation completely leakage-free.

```
         Fan End                Drive End
    ┌──────────────────────────────────────┐
    │   I     O     B  │  I     O     B    │
 7  │  TRAIN TRAIN TEST│ TRAIN TRAIN TRAIN │
14  │  TEST  TRAIN TRAIN TEST  TEST  TRAIN │
21  │  TRAIN TEST  TRAIN TRAIN TRAIN TEST  │
    └──────────────────────────────────────┘
         (example split — randomized each run)
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-username/cwru-multi-label.git
cd cwru-multi-label
pip install pandas
```

### Usage

```python
from multi_label_cwru import MultiLabelSplitter

# 1. Define paths
cwru_root   = "/path/to/CWRU"
output_path = "/path/to/multi_label_cwru"

# 2. Initialize
splitter = MultiLabelSplitter(cwru_root)

# 3. Create output directory structure
splitter.create_folders(output_path)

# 4. Generate train / test split
train, test = splitter.filtering()

# 5. Copy files to output directories
splitter.splitting(output_path, train, train=True)
splitter.splitting(output_path, test,  train=False)

# 6. Move healthy (Normal) samples to test set
splitter.move_normal(output_path)
```

See [`example_usage.ipynb`](https://claude.ai/chat/example_usage.ipynb) for a complete walkthrough.

---

## 📋 API Reference

### `MultiLabelSplitter(cwru_root_path)`

| Parameter          | Type    | Description                        |
| ------------------ | ------- | ---------------------------------- |
| `cwru_root_path` | `str` | Root directory of the CWRU dataset |

---

### `.create_folders(output_path)`

Creates the full output directory tree for all fault types, locations, and splits.

| Parameter       | Type    | Description                                |
| --------------- | ------- | ------------------------------------------ |
| `output_path` | `str` | Destination root for the organized dataset |

---

### `.filtering() → (train_df, test_df)`

Scans the CWRU root, applies the random fault-size split recipe, and returns two Pandas DataFrames.

```
DataFrame columns:
  path      — absolute path to .mat file
  label     — multi-label vector, e.g. [1, 0, 0]
  type      — fault type: 'IR', 'OR', or 'B'
  severity  — fault size: '007', '014', or '021'
  location  — accelerometer location: 'DE' or 'FE'
```

**Example output:**

```
Split Recipe
------------------------------
DE-IR -> 021
DE-OR -> 007
DE-B  -> 007
FE-IR -> 021
FE-OR -> 014
FE-B  -> 007

Dataset Summary
------------------------------
Train: 45
Test : 21
```

---

### `.splitting(output_path, df, train=True)`

Copies `.mat` files to the appropriate `train/` or `test/` subdirectory.

| Parameter       | Type          | Description                                          |
| --------------- | ------------- | ---------------------------------------------------- |
| `output_path` | `str`       | Root of the output directory                         |
| `df`          | `DataFrame` | Train or test DataFrame from `.filtering()`        |
| `train`       | `bool`      | `True`→ copies to `train/`,`False`→`test/` |

---

### `.move_normal(output_path)`

Copies all healthy bearing `.mat` files to `Normal/test/`. Healthy signals are **never** used for training.

| Parameter       | Type    | Description                  |
| --------------- | ------- | ---------------------------- |
| `output_path` | `str` | Root of the output directory |

---
