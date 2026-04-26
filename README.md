<div align="center">

# 🧠 ML Lab Programs

*Machine learning algorithms built from scratch — to understand the math, not just the API.*

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=matplotlib&logoColor=white)

</div>

---

## 📂 Algorithms

### 🔍 Concept Learning

| Algorithm | File | What it does |
|-----------|------|-------------|
| **Find-S** | `find_s.py` | Finds the most specific hypothesis consistent with all positive examples |
| **Candidate Elimination** | `candidate_elimination.py` | Maintains the full version space — both S and G boundaries — over all examples |

### 🏷️ Classification

| Algorithm | File | What it does |
|-----------|------|-------------|
| **Gaussian Naive Bayes** | `naive_bayes_gaussian.py` | Computes class posteriors using per-feature Gaussian likelihoods — no sklearn |
| **Decision Tree** | `decision_tree.py` | Entropy-based splitting via sklearn with full tree visualization |
| **Neural Network** | `ann.py` | 2-layer feedforward net with sigmoid + softmax, backprop by hand |

### 📈 Regression

| Algorithm | File | What it does |
|-----------|------|-------------|
| **Linear Regression** | `linear_regression.py` | Closed-form solution via Moore-Penrose pseudoinverse: `w = (XᵀX)⁻¹Xᵀy` |

### 🔵 Clustering

| Algorithm | File | What it does |
|-----------|------|-------------|
| **K-Means** | `k_means_clustering.py` | Iterative centroid clustering with convergence detection and 2D visualization |

---

## ⚙️ Shared Pipeline

Every algorithm follows the same flow:

```
CSV  ──▶  X / y split  ──▶  Normalize  ──▶  80/20 split  ──▶  Train  ──▶  Evaluate
```

> `random_state=42` · `test_size=0.2` · label always in the last column

---

## 🤖 Neural Network Architecture

Implemented end-to-end in NumPy — no frameworks.

```
┌─────────────┐     sigmoid     ┌─────────────┐     softmax     ┌─────────────┐
│   Input     │ ─────────────▶ │   Hidden    │ ─────────────▶ │   Output    │
│   4 units   │                 │   8 units   │                 │   2 units   │
└─────────────┘                 └─────────────┘                 └─────────────┘
```

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | `0.1` |
| Epochs | `1000` |
| Weight init | Xavier (`√1/n`) |
| Loss | Cross-entropy |
| Optimizer | Vanilla gradient descent |

---

## 🗃️ Datasets

<details>
<summary><strong>dataset.csv</strong> — Student academic performance (40 samples)</summary>

Used by: Naive Bayes, Decision Tree, Linear Regression, K-Means, ANN

| Feature | Description |
|---------|-------------|
| `math_score` | Math exam score |
| `science_score` | Science exam score |
| `english_score` | English exam score |
| `attendance_pct` | Attendance percentage |
| `label` | Pass `1` / Fail `0` |

</details>

<details>
<summary><strong>data.csv</strong> — EnjoySport concept learning dataset (8 samples)</summary>

Used by: Find-S, Candidate Elimination

| Feature | Values |
|---------|--------|
| `Sky` | Sunny, Rainy, Cloudy |
| `AirTemp` | Warm, Cold, Hot |
| `Humidity` | Normal, High |
| `Wind` | Strong, Weak |
| `Water` | Warm, Cool |
| `Forecast` | Same, Change |
| `EnjoySport` | yes / no |

</details>

---

## 🚀 Run

```bash
python find_s.py
python candidate_elimination.py
python naive_bayes_gaussian.py
python decision_tree.py
python linear_regression.py
python k_means_clustering.py
python ann.py
```
