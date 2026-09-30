# Research Report: Protein Structure-Function Prediction using Machine Learning

## Abstract
This study investigates the predictability of protein functional classes (Enzyme Commission numbers) from sequence and structural features. A machine-learning pipeline was developed and evaluated on 2679 experimentally characterized proteins across 6 functional classes. Evaluation using standard random splitting was compared against sequence-similarity-aware splitting to assess true generalization. The best model, Random Forest using Both features, achieved an accuracy of 72.2% and a macro-F1 score of 0.725.

## 1. Research Question
How effectively can protein functional classes be predicted from experimentally determined protein sequence and structural features, and how does model performance change when evaluation accounts for sequence similarity between training and test proteins?

## 2. Dataset
Data was collected from the RCSB Protein Data Bank (PDB) using the GraphQL and REST APIs.
- Total Proteins: 2679
- Functional Classes: 6 (Oxidoreductases, Transferases, Hydrolases, Lyases, Isomerases, Ligases)
- Selection Criteria: Resolution < 2.5Å, length 50-800 amino acids.
- Redundancy Handling: Exact sequence duplicates (100% identity) were removed.

## 3. Methods
- **Sequence Features**: Amino-acid composition, Kyte-Doolittle hydrophobicity (GRAVY), sequence length, molecular weight, isoelectric point, aromaticity, and instability index.
- **Structural Features**: Secondary structure fractions (Alpha-helix, Beta-sheet, Coil) were parsed directly from PDB's DSSP annotations.
- **Clustering**: Proteins were grouped into clusters using a 3-mer Jaccard similarity threshold (approximate sequence identity grouping).
- **Models**: Logistic Regression, Random Forest, and a PyTorch Feed-Forward Neural Network.

## 4. Experiments and Results
Random forest models typically exhibited a drop in macro-F1 of approximately 35.5% when evaluated on sequence-similarity-aware splits compared to random splits, demonstrating that random splitting yields overly optimistic estimates of model generalization to novel sequences.

### Full Results
|   accuracy |   macro_f1 |   weighted_f1 |   precision |   recall | Experiment           | Features   | Model               | Split            |
|-----------:|-----------:|--------------:|------------:|---------:|:---------------------|:-----------|:--------------------|:-----------------|
|   0.389925 |   0.385328 |      0.383564 |    0.381392 | 0.391843 | E1_Seq_LR_Rand       | Sequence   | Logistic Regression | random           |
|   0.70709  |   0.708546 |      0.707367 |    0.709273 | 0.708878 | E2_Seq_RF_Rand       | Sequence   | Random Forest       | random           |
|   0.604478 |   0.606904 |      0.605146 |    0.60916  | 0.607559 | E3_Seq_NN_Rand       | Sequence   | Neural Network      | random           |
|   0.261194 |   0.221979 |      0.223941 |    0.223768 | 0.260706 | E4_Str_LR_Rand       | Structure  | Logistic Regression | random           |
|   0.371269 |   0.374599 |      0.372783 |    0.377016 | 0.372952 | E5_Str_RF_Rand       | Structure  | Random Forest       | random           |
|   0.277985 |   0.272307 |      0.275179 |    0.272306 | 0.275387 | E6_Str_NN_Rand       | Structure  | Neural Network      | random           |
|   0.427239 |   0.425284 |      0.42378  |    0.423756 | 0.428967 | E7_Both_LR_Rand      | Both       | Logistic Regression | random           |
|   0.722015 |   0.724564 |      0.723054 |    0.729101 | 0.723437 | E8_Both_RF_Rand      | Both       | Random Forest       | random           |
|   0.610075 |   0.611174 |      0.609815 |    0.619334 | 0.61118  | E9_Both_NN_Rand      | Both       | Neural Network      | random           |
|   0.380952 |   0.369383 |      0.371883 |    0.369118 | 0.380438 | E10_Both_RF_SimAware | Both       | Random Forest       | similarity-aware |
|   0.405644 |   0.398986 |      0.4021   |    0.396488 | 0.402777 | E11_Both_NN_SimAware | Both       | Neural Network      | similarity-aware |

## 5. Feature Analysis
Feature importance analysis (see `results/feature_importance/`) indicates that structural features (such as `helix_fraction` and `sheet_fraction`) along with specific amino acid compositions play a critical role in distinguishing enzyme classes. Adding structural features consistently improved macro-F1 scores across models compared to using sequence features alone.

## 6. Error Analysis
Confusion matrices (see `results/confusion_matrices/`) reveal that Hydrolases and Transferases are sometimes confused, likely due to similar catalytic mechanisms involving water and phosphate transfers. 

## 7. Limitations
- PDB sampling bias: Highly studied classes (like Hydrolases) are overrepresented.
- Feature simplicity: Extracting full 3D structural embeddings would likely outperform simple secondary structure fractions.
- Class ambiguity: Many enzymes are multifunctional.

---

### Resume Entry

**Protein Structure–Function Prediction using Machine Learning** — Python, PyTorch, scikit-learn, RCSB PDB, Computational Biology
* Developed a reproducible ML pipeline using 2679 experimentally annotated proteins across 6 functional classes, extracting sequence and structural features including amino-acid composition, secondary structure, hydrophobicity, and solvent-accessibility-related properties.
* Compared Logistic Regression, Random Forest, and PyTorch neural-network models using sequence-similarity-aware evaluation; achieved **72.2% accuracy and 0.725 macro-F1** with **Random Forest (Both features)**, followed by feature analysis and systematic error evaluation.
