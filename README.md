# Mushroom Edibility Classification (CLI-Based Supervised Machine Learning)

A CLI-based supervised machine learning application that classifies mushrooms as **Edible (e)** or **Poisonous (p)** using the UCI Mushroom dataset. The project provides a complete, leakage-aware ML workflow implemented in Jupyter Notebook and a command-line interface for deployment-style predictions.

---
## Repository Structure
```
│
├── Data/
│ ├── agaricus-lepiota.data
│ └── agaricus-lepiota.names
│
├── notebooks/
│ └── MushroomEdibilityPrediction.ipynb
│
├── reports/
│ └── 23049051 Digdarshan Bhattarai.pdf
│
```

---
## Project Purpose

Misclassifying a poisonous mushroom as edible is a safety-critical error. This project builds and evaluates supervised classification models that predict mushroom edibility from categorical physical attributes. Model selection and interpretation emphasise **Recall for the poisonous class (p)** to reduce the risk of poisonous-to-edible misclassification.

---

## What This Project Does

### 1) Data Preparation and Quality Assurance
- Loads the UCI Mushroom dataset and assigns a consistent column schema.
- Converts missing markers (`?`) into a usable representation and handles missingness.
- Removes non-informative feature(s) (e.g., constant columns).
- Performs dataset checks (missing values, basic integrity verification).
- Examines class distribution to support evaluation decisions.

### 2) Leakage-Safe Training and Tuning
- Uses a **scikit-learn Pipeline** to ensure preprocessing is fitted only on training data.
- Applies **OneHotEncoder** inside the pipeline to handle categorical features safely.
- Uses a **stratified train/test split** to preserve class proportions in evaluation.
- Performs **GridSearchCV** with **StratifiedKFold** cross-validation.
- Optimises hyperparameters using **Recall(p)** as the primary scoring objective.

### 3) Models Implemented
- **Logistic Regression**
- **Bernoulli Naïve Bayes**
- **Random Forest**

### 4) Evaluation and Comparison
- Reports hold-out test metrics: **Accuracy, Precision(p), Recall(p), F1(p)**
- Generates **confusion matrices** to inspect error types (especially poisonous predicted as edible).
- Produces a consolidated comparison table and metric visualisations.
- Supports deployment-style inference through a CLI application.

---

