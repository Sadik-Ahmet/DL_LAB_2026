# YZM304 Derin Öğrenme – I. Proje Ödevi
## Binary Classification with Breast Cancer Wisconsin Dataset

**Ankara Üniversitesi | Yapay Zeka ve Veri Mühendisliği | 2025-2026 Bahar Dönemi**

---

## Introduction

This project implements a multi-layer binary classification neural network from scratch using NumPy, then replicates the same architecture using Scikit-learn's `MLPClassifier` and PyTorch. The dataset used is the **Breast Cancer Wisconsin** dataset (`sklearn.datasets.load_breast_cancer`), which contains 569 samples described by 30 numerical features, classified as either *Malignant* (0) or *Benign* (1).

The primary motivations are:
1. Understanding the internal mechanics of backpropagation and gradient descent by building without deep learning frameworks.
2. Validating the NumPy implementation against established libraries (Scikit-learn, PyTorch) using identical hyperparameters.
3. Studying overfitting/underfitting behavior through learning curves and train/dev/test metric analysis.
4. Applying model selection based on a quantitative criterion (steps to ≥90% validation accuracy).

The binary nature of the task (malignant vs. benign tumor) makes Binary Cross-Entropy (BCE) loss and sigmoid output activation the natural choices. This dataset is well-studied, balanced (357 benign, 212 malignant), and well-suited for demonstrating neural network fundamentals.

---

## Methods

### Dataset & Preprocessing

| Property | Value |
|---|---|
| Dataset | Breast Cancer Wisconsin (`sklearn.datasets`) |
| Samples | 569 |
| Features | 30 (continuous, no missing values) |
| Classes | Malignant (0) = 212, Benign (1) = 357 |
| Preprocessing | StandardScaler (zero mean, unit variance) |
| Train split | 70% (~398 samples) |
| Dev split | 15% (~85 samples) |
| Test split | 15% (~86 samples) |
| Stratified | Yes (`stratify=y`) |
| Random seed | 42 |

### Global Hyperparameters

| Hyperparameter | Value |
|---|---|
| Learning rate (`lr`) | 0.01 |
| Training epochs (`n_steps`) | 1000 |
| Optimizer | SGD + Momentum (PyTorch) / SGD (NumPy) |
| Loss function | Binary Cross-Entropy (BCE) |
| Hidden activation | ReLU |
| Output activation | Sigmoid |
| Weight initialization | He initialization (scale = √(2/fan_in)) |
| Random seed | 42 |
| Model selection threshold | ≥ 90% dev accuracy |

### Regularization & Optimization Techniques

| Technique | Detail | Applied In |
|---|---|---|
| **L2 Weight Decay** | λ ∈ {0.0, 0.001, 0.01, 0.1} sweep | NumPy NN (Section 4.5) |
| **L2 Regularization** | alpha = 0.001 | Scikit-learn MLPClassifier |
| **L2 Weight Decay** | weight_decay = 1e-3 | PyTorch SGD optimizer |
| **Mini-Batch SGD** | batch_size = 32, shuffle=True | PyTorch DataLoader |
| **Mini-Batch SGD** | batch_size = 32 | Scikit-learn MLPClassifier |
| **Momentum** | momentum = 0.9 | PyTorch SGD optimizer |

The L2 penalty term added to Binary Cross-Entropy is:

$$\mathcal{L}_{total} = \mathcal{L}_{BCE} + \frac{\lambda}{2m} \sum_{l} \|W^{(l)}\|_F^2$$

This penalizes large weights, reducing model complexity and variance (overfitting).

### Model Architectures

| Model | Architecture | Description |
|---|---|---|
| **M1 (Lab Model)** | 30 → 16 → 1 | 1 hidden layer (lab baseline) |
| **M2** | 30 → 32 → 16 → 1 | 2 hidden layers (more capacity) |
| **M3** | 30 → 64 → 32 → 16 → 1 | 3 hidden layers (deeper model) |

### NeuralNetwork Class (NumPy)

The `NeuralNetwork` class encapsulates the full training loop:

- **Constructor** `__init__(layer_sizes, lr, n_steps, seed)`: stores configuration, calls `__init_weights()`.
- **Private methods**: `__init_weights`, `__forward`, `__backward`, `__update_params`, `__compute_loss`, `__relu`, `__sigmoid`.
- **Public methods**: `fit(X_train, y_train, X_dev, y_dev)`, `predict(X)`, `predict_proba(X)`, `evaluate(X, y)`.

### Model Selection Criterion

Among models whose dev accuracy reaches ≥ 90%, the one reaching this threshold in the **fewest training steps** is selected. If none reach the threshold, the model with the highest dev accuracy is chosen.

### Library Replications

Both Scikit-learn and PyTorch implementations use:
- Identical hidden layer architecture as the selected NumPy model
- Same SGD optimizer (`lr = 0.01`)
- Same random seed (42)
- BCE loss function

---

## Results

### Overfitting / Underfitting Analysis

| Model | Train Acc | Dev Acc | Test Acc | Status |
|---|---|---|---|---|
| M1 (Lab) [30→16→1] | ~0.975 | ~0.977 | ~0.930 | Good Fit |
| M2 [30→32→16→1] | ~0.982 | ~1.000 | ~0.942 | Good Fit |
| M3 [30→64→32→16→1] | ~0.985 | ~0.988 | ~0.930 | Good Fit |

*Note: Exact values depend on execution; see notebook output.*

**Selected model:** M2 — reaches 90% dev accuracy in fewest steps.

### Regularization Experiment (L2 Weight Decay Sweep — M2 Architecture)

To study bias-variance trade-off, L2 weight decay was applied at four different λ values:

| λ (weight_decay) | Train Acc | Dev Acc | Test Acc | Bias/Variance Diagnosis |
|---|---|---|---|---|
| 0.000 (no reg.) | ~0.982 | ~1.000 | ~0.942 | İyi Fit |
| 0.001 | ~0.975 | ~0.977 | ~0.942 | İyi Fit (variance ↓) |
| 0.010 | ~0.968 | ~0.965 | ~0.930 | İyi Fit (slight bias ↑) |
| 0.100 | ~0.940 | ~0.930 | ~0.920 | High Bias (underfitting) |

*Exact values shown in notebook Section 4.5 output and `regularization_analysis.png`.*

**Key finding:** λ = 0.001 provides the best bias-variance balance — it slightly reduces the train-dev gap compared to no regularization while maintaining high test accuracy. λ = 0.1 induces underfitting (high bias), demonstrating that over-regularization is as harmful as under-regularization.

### Test Set Metrics (all three implementations)

| Implementation | Accuracy | Precision | Recall | F1 | Regularization |
|---|---|---|---|---|---|
| NumPy NN (M2) | ~0.942 | ~0.962 | ~0.944 | ~0.953 | L2 λ=0.001 |
| Scikit-learn MLP | ~0.942 | ~0.980 | ~0.926 | ~0.952 | L2 α=0.001, batch=32 |
| PyTorch NN | ~0.942 | ~0.962 | ~0.944 | ~0.953 | L2 wd=1e-3, batch=32 |

*Note: Small differences between implementations stem from internal optimizer precision and weight initialization ordering.*

### Generated Plots

| File | Description |
|---|---|
| `eda_plots.png` | Class distribution, feature means, correlation heatmap, boxplots |
| `learning_curves.png` | BCE loss & accuracy curves for M1, M2, M3 |
| `regularization_analysis.png` | L2 λ sweep: Train/Dev/Test accuracy & variance analysis |
| `pytorch_curves.png` | PyTorch loss & accuracy curves (mini-batch SGD) |
| `confusion_matrices.png` | Confusion matrices for all 3 implementations |
| `model_comparison.png` | Bar chart comparing Accuracy/Precision/Recall/F1 |

---

## Discussion

### Key Findings

- All three models (M1, M2, M3) show **no significant overfitting** — the gap between train and dev accuracy is very small (<3%), which is expected given the size and quality of the Breast Cancer dataset.
- **M2 (2 hidden layers)** was selected as the best model: it achieves excellent dev accuracy while reaching the 90% threshold in fewer steps than M3, demonstrating that deeper is not always better.
- The **NumPy, Scikit-learn, and PyTorch** implementations produce nearly identical results (~94% accuracy), confirming the correctness of the hand-coded backpropagation.
- **StandardScaler** preprocessing significantly helps convergence; without normalization, gradient magnitudes vary wildly across features.

### Bias–Variance Trade-off & Regularization Analysis

The primary tool for addressing high variance (overfitting) in this work is **L2 weight decay**. Results from Section 4.5 confirm:

- **λ = 0.0 (no regularization):** Baseline. All models already fit well on this dataset, so overfitting is mild. The train-dev gap for M2 is < 3%.
- **λ = 0.001:** Optimal point — slightly narrows the train-dev gap, maintaining test accuracy. This is the value applied in all three library implementations.
- **λ = 0.01:** Marginally reduces both train and dev accuracy, indicating the regularization begins to dominate.
- **λ = 0.1:** Induces **high bias (underfitting)** — the model can no longer fit the training data well, demonstrating the classic bias-variance trade-off visually.

This experiment directly demonstrates the assignment requirement: *"yüksek varyans ya da bias problemleri ele alınarak regülarizasyon teknikleri kullanılmalıdır."*

### Mini-Batch SGD

The PyTorch and Scikit-learn implementations use **mini-batch SGD** with `batch_size = 32`. Compared to full-batch gradient descent (used in the NumPy implementation for clarity of implementation):

- Mini-batch SGD introduces **noise into gradient estimates**, which acts as an implicit regularizer.
- It allows the optimizer to escape sharp local minima, often yielding better generalization.
- With `momentum = 0.9` (PyTorch), updates are smoothed across batches, accelerating convergence.

### Limitations & Future Work

- **Dropout regularization**: L2 was applied in this work; Dropout (randomly zeroing neurons during training) was not, and could provide complementary regularization.
- **Hyperparameter search**: Learning rate and λ were chosen by manual sweep; a systematic grid or random search could find globally optimal configurations.
- **Data augmentation**: The dataset is relatively small (569 samples). Synthetic data generation (e.g., SMOTE for class balance) could be explored.
- **Multi-class extension**: Adding a Softmax output layer and categorical cross-entropy loss would generalize this framework to datasets with more than 2 classes.

---

## Project Structure

```
Homework_1_DL/
├── DL_Homework_1_YZM304.ipynb   # Main Jupyter Notebook (all sections)
├── README.md                    # This file (IMRAD format)
├── requirements.txt             # Python dependencies
├── generate_notebook.py         # Script used to build the .ipynb
└── YZM304_Proje_Odevi1_2526-1.pdf  # Original assignment PDF
```

---

## Reproducibility

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Jupyter
jupyter notebook DL_Homework_1_YZM304.ipynb

# 3. Run All Cells (Kernel → Restart & Run All)
```

All random states are fixed via `SEED = 42`. Re-running the notebook end-to-end should reproduce identical results.

---

*Ankara Üniversitesi – YZM304 Derin Öğrenme | 2025-2026 Bahar Dönemi*
