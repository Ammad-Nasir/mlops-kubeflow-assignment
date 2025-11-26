## MLOps Kubeflow / MLflow Assignment

This repository contains the solution for the **Cloud MLOps Assignment 4** using:
- **DVC** for data versioning
- **Kubeflow Pipelines (KFP)** for pipeline orchestration (with optional **MLflow** for tracking if KFP is unavailable)

The core ML problem is **regression on the Boston Housing dataset** using a **Random Forest** model.

### Project Overview

- **Data**: Boston housing dataset, stored as a CSV at `data/raw/boston_housing.csv` and versioned with DVC.
- **Pipeline steps**:
  - Data extraction (from DVC-managed dataset)
  - Data preprocessing (scaling and train/test split)
  - Model training (RandomForestRegressor)
  - Model evaluation (MSE, R²)
- **MLOps Tools**:
  - Git + GitHub
  - DVC
  - Kubeflow Pipelines on Minikube
  - Jenkins / GitHub Actions for CI

### Setup Instructions

- **Python environment**
  - Create and activate a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

- **DVC initialization and data versioning**
  1. Initialize DVC in the repo:

```bash
dvc init
```

  2. Generate the Boston housing CSV (uses `load_boston` from scikit-learn):

```bash
python prepare_data.py
```

  3. Track the dataset with DVC:

```bash
dvc add data/raw/boston_housing.csv
```

  4. Configure a DVC remote (example: local folder):

```bash
dvc remote add -d local_remote /path/to/dvc_remote_folder
```

  5. Push data to the remote:

```bash
dvc push
```

- **Kubeflow Pipelines on Minikube (high level)**
  - Start Minikube:

```bash
minikube start
```

  - Deploy KFP (standalone or full Kubeflow; follow official docs).
  - Access KFP UI (usually `http://localhost:8080` or via `kubectl port-forward`).

- **Compile and upload pipeline**

```bash
python pipeline.py            # generates pipeline.yaml
```

Then upload `pipeline.yaml` via the Kubeflow Pipelines UI and run a new experiment.

### Pipeline Walkthrough

1. `data_extraction_component` reads the DVC-managed CSV (assumed to be fetched via `dvc pull` or baked into the image).
2. `data_preprocessing_component` scales features and creates train/test splits, saving them as NumPy arrays.
3. `model_training_component` trains a Random Forest model and stores it under `models/random_forest.joblib`.
4. `model_evaluation_component` evaluates the model on the test set and writes metrics (`mse`, `r2`) to `artifacts/metrics.json`.

The `pipeline.py` file wires these components together into a single Kubeflow pipeline and can be compiled to `pipeline.yaml` for KFP.



