# 🧬 Protein Structure-Function Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge\&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge\&logo=scikit-learn\&logoColor=white)
![Biopython](https://img.shields.io/badge/Biopython-8CAE42?style=for-the-badge)

A comprehensive and reproducible machine-learning pipeline for predicting the functional class of experimentally characterized proteins using **sequence composition** and **structural secondary features** extracted from the **RCSB Protein Data Bank (PDB)**.

The project combines classical machine-learning models with a PyTorch neural network and includes **sequence-similarity-aware evaluation** to reduce data leakage caused by highly homologous protein sequence.

---

## 📑 Table of Contents

* [Overview](#-overview)
* [Dataset](#-dataset)
* [Machine Learning Pipeline](#-machine-learning-pipeline)
* [Results & Evaluation](#-results--evaluation)
* [Getting Started](#-getting-started)
* [Project Structure](#-project-structure)
* [License](#-license)

---

## 🔬 Overview

Enzymes play a critical role in catalyzing biological reactions. Predicting their functional class from sequence and structural information is an important problem in computational biology.

This project builds a machine-learning pipeline using **2,679 experimentally annotated proteins** across **6 major Enzyme Commission (EC) classes**.

The pipeline evaluates:

* Logistic Regression
* Random Forest
* PyTorch Feed-Forward Neural Network (MLP)

A key feature of this project is **sequence-similarity-aware evaluation**, which helps prevent overly optimistic performance caused by highly similar sequences appearing in both training and testing sets.

### 🎯 Main Objective

Predict the major EC functional class of a protein using:

1. Sequence-derived physicochemical features
2. Amino-acid composition
3. Structural secondary-structure features
4. Machine-learning models

---

## 📊 Dataset

Protein data is programmatically collected from the **RCSB Protein Data Bank (PDB)** using its GraphQL and REST APIs.

### Dataset Summary

| Property                 | Description               |
| ------------------------ | ------------------------- |
| **Total Proteins**       | ~2,679 unique sequences   |
| **Target Variable**      | Major EC functional class |
| **Sequence Length**      | 50–800 amino acids        |
| **Structure Resolution** | < 2.5 Å                   |
| **Data Source**          | RCSB Protein Data Bank    |

### 🧬 EC Functional Classes

The dataset contains six major enzyme classes:

1. **Oxidoreductases**
2. **Transferases**
3. **Hydrolases**
4. **Lyases**
5. **Isomerases**
6. **Ligases**

---

## ⚙️ Machine Learning Pipeline

The complete workflow is automated through a master orchestration script.

### 1. 📥 Data Collection

Protein sequences and structural information are fetched programmatically from the RCSB PDB APIs.

The collection process retrieves:

* Protein sequences
* Experimental structure information
* EC annotations
* Secondary-structure information

---

### 2. 🧬 Feature Extraction

#### Sequence Features

Sequence-level physicochemical properties are calculated using **Biopython**, including:

* Molecular weight
* Isoelectric point
* Kyte-Doolittle hydrophobicity (GRAVY)
* Instability index
* Amino-acid composition

#### Structural Features

Structural features are extracted from PDB information, including normalized secondary-structure fractions such as:

* Alpha-helix fraction
* Beta-sheet fraction
* Coil/other structure fraction

---

### 3. 🧹 Preprocessing

The preprocessing stage:

* Maps proteins to their major EC classes
* Handles missing values
* Combines sequence and structural features
* Prepares the final feature matrix
* Balances the dataset where required

---

### 4. 🔗 Similarity Clustering

To reduce data leakage, protein sequences are grouped according to **k-mer Jaccard similarity**.

This enables the creation of similarity-aware train/test splits where highly similar sequences are less likely to appear across both sets.

This provides a more realistic evaluation of model generalization to previously unseen protein sequences.

---

### 5. 🤖 Model Training

Three machine-learning approaches are evaluated:

| Model                   | Description                    |
| ----------------------- | ------------------------------ |
| **Logistic Regression** | Linear classification baseline |
| **Random Forest**       | Ensemble of decision trees     |
| **PyTorch MLP**         | Feed-forward neural network    |

The models are evaluated using:

* Sequence features
* Structural features
* Combined sequence + structural features

Both random and similarity-aware splits are tested.

---

## 🏆 Results & Evaluation

The experiments compare conventional random splitting with sequence-similarity-aware evaluation.

The similarity-aware evaluation produces lower performance than random splitting, highlighting the effect of sequence redundancy and potential leakage in conventional evaluation.

### 🥇 Best Random-Split Configuration

The highest reported random-split performance was obtained using:

**Random Forest + Sequence & Structure Features**

| Metric       |      Score |
| ------------ | ---------: |
| **Accuracy** | **72.20%** |
| **Macro-F1** | **0.7246** |

### 📋 Full Experiment Results

| Experiment           | Features  | Model               | Split            |   Accuracy |   Macro-F1 |
| -------------------- | --------- | ------------------- | ---------------- | ---------: | ---------: |
| E1_Seq_LR_Rand       | Sequence  | Logistic Regression | Random           |     38.99% |     0.3853 |
| E2_Seq_RF_Rand       | Sequence  | Random Forest       | Random           |     70.71% |     0.7085 |
| E3_Seq_NN_Rand       | Sequence  | Neural Network      | Random           |     62.50% |     0.6264 |
| E4_Str_LR_Rand       | Structure | Logistic Regression | Random           |     26.12% |     0.2220 |
| E5_Str_RF_Rand       | Structure | Random Forest       | Random           |     37.13% |     0.3746 |
| E6_Str_NN_Rand       | Structure | Neural Network      | Random           |     30.22% |     0.2987 |
| E7_Both_LR_Rand      | Both      | Logistic Regression | Random           |     42.72% |     0.4253 |
| **E8_Both_RF_Rand**  | **Both**  | **Random Forest**   | **Random**       | **72.20%** | **0.7246** |
| E9_Both_NN_Rand      | Both      | Neural Network      | Random           |     61.38% |     0.6170 |
| E10_Both_RF_SimAware | Both      | Random Forest       | Similarity-aware |     38.10% |     0.3694 |
| E11_Both_NN_SimAware | Both      | Neural Network      | Similarity-aware |     34.04% |     0.3337 |

### 📈 Evaluation Visualizations

The repository includes generated visualizations in the `results/` directory:

* Confusion matrices
* Feature importance plots
* Aggregate experiment metrics

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Krushna018/Protein-Structure-Function-Prediction-using-Machine-Learning.git

cd Protein-Structure-Function-Prediction-using-Machine-Learning
```

### 2. Install Dependencies

Make sure Python 3.8+ is installed.

```bash
pip install -r requirements.txt
pip install tabulate
```

### 3. Run the Complete Pipeline

The entire workflow can be executed using:

```bash
python run_all.py
```

The pipeline automatically performs:

```text
Data Collection
      ↓
Feature Extraction
      ↓
Preprocessing
      ↓
Similarity Clustering
      ↓
Model Training
      ↓
Evaluation
      ↓
Report Generation
```

> **Note:** By default, data collection may be skipped when `data/raw/dataset.csv` already exists. To perform a fresh RCSB PDB data collection, enable the corresponding data-collection step in `run_all.py`.

---

## 📁 Project Structure

```text
Protein-Structure-Function-Prediction-using-Machine-Learning/
│
├── data/
│   ├── raw/
│   │   └── dataset.csv
│   └── processed/
│       └── feature matrices and similarity clusters
│
├── experiments/
│   └── run_experiments.py
│
├── report/
│   └── research_report.md
│
├── results/
│   ├── confusion_matrices/
│   ├── feature_importance/
│   └── metrics/
│
├── src/
│   ├── data_collection.py
│   ├── sequence_features.py
│   ├── structure_features.py
│   ├── preprocessing.py
│   ├── similarity_clustering.py
│   ├── models.py
│   ├── training.py
│   └── generate_report.py
│
├── run_all.py
├── requirements.txt
└── README.md
```

### 📦 Core Modules

| Module                     | Purpose                                        |
| -------------------------- | ---------------------------------------------- |
| `data_collection.py`       | Fetches protein data from RCSB PDB             |
| `sequence_features.py`     | Extracts physicochemical and sequence features |
| `structure_features.py`    | Extracts secondary-structure features          |
| `preprocessing.py`         | Cleans and prepares the dataset                |
| `similarity_clustering.py` | Performs k-mer similarity clustering           |
| `models.py`                | Defines ML and neural-network models           |
| `training.py`              | Handles model training and evaluation          |
| `generate_report.py`       | Generates the experiment report                |
| `run_all.py`               | Runs the complete pipeline                     |

---

## 🔬 Reproducibility

The project is designed to make the complete experimental workflow reproducible.

A single command can execute the major stages of the project:

```bash
python run_all.py
```

Generated outputs include:

* Processed datasets
* Model metrics
* Confusion matrices
* Feature-importance visualizations
* Research report

---
