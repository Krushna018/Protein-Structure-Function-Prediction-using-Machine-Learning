# Protein Structure-Function Prediction using Machine Learning

This project implements a reproducible machine-learning pipeline to predict the functional class (Enzyme Commission numbers) of experimentally characterized proteins using both sequence and structural information.

## Project Structure

* `src/`: Core Python modules for data collection, feature extraction, preprocessing, and modeling.
* `experiments/`: Scripts that define and run the machine learning experiments.
* `data/raw/`: The dataset is downloaded and stored here.
* `data/processed/`: Preprocessed and clustered datasets are saved here.
* `results/`: Contains output metrics (CSV), confusion matrices (PNG), and feature importance plots (PNG).
* `report/`: The final research report and metrics synthesis is generated here.
* `run_all.py`: The orchestrator script to execute the entire pipeline end-to-end.

## How to Run

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required Python packages:

```bash
pip install -r requirements.txt
pip install tabulate
```

### 2. Execute the Pipeline
To run the entire pipeline from data collection through model training to report generation, run the orchestrator script from the project root:

```bash
python run_all.py
```

*Note: In `run_all.py`, the data collection step (`python -u src/data_collection.py`) might be commented out if the data has already been fetched to save time. You can uncomment it to fetch fresh data from the RCSB PDB GraphQL API.*

### Pipeline Steps (Executed automatically by `run_all.py`)
1. **Data Collection (`src/data_collection.py`)**: Fetches ~2,600 protein sequences and metadata across 6 EC classes from the PDB.
2. **Sequence Features (`src/sequence_features.py`)**: Computes physicochemical properties (molecular weight, isoelectric point, amino acid composition, etc.) using Biopython.
3. **Structure Features (`src/structure_features.py`)**: Computes secondary structure fractions (helix, sheet, coil) based on PDB instance features.
4. **Preprocessing (`src/preprocessing.py`)**: Merges features, maps EC classes, removes NaNs, and splits the data.
5. **Clustering (`src/similarity_clustering.py`)**: Groups sequences based on k-mer Jaccard similarity for non-redundant train/test splitting.
6. **Experiments (`experiments/run_experiments.py`)**: Trains and evaluates Logistic Regression, Random Forest, and PyTorch MLPs on different feature sets and data splits.
7. **Report Generation (`src/generate_report.py`)**: Aggregates the results and generates `report/research_report.md`.
