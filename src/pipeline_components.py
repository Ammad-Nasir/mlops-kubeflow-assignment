import os
import json

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from kfp import dsl
from kfp.v2.dsl import OutputPath  # <-- important

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_and_save_california_csv(output_path: str) -> str:
    """Load California housing dataset and save as CSV."""
    _ensure_dir(os.path.dirname(output_path))
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    df.to_csv(output_path, index=False)
    return output_path

@dsl.component(base_image="python:3.10-slim")
def data_extraction_component(output_csv_path: str) -> str:
    """Ensure dataset exists at output_csv_path."""
    if not os.path.exists(output_csv_path):
        load_and_save_california_csv(output_csv_path)
    return output_csv_path

@dsl.component(base_image="python:3.10-slim")
def data_preprocessing_component(
    input_csv_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    output_train_path: OutputPath('NPY') = "data/processed/train.npy",
    output_test_path: OutputPath('NPY') = "data/processed/test.npy",
):
    os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_test_path), exist_ok=True)

    df = pd.read_csv(input_csv_path)
    X = df.drop("MedHouseVal", axis=1).values  # target column in California dataset
    y = df["MedHouseVal"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    np.save(output_train_path, {"X": X_train_scaled, "y": y_train})
    np.save(output_test_path, {"X": X_test_scaled, "y": y_test})

@dsl.component(base_image="python:3.10-slim")
def model_training_component(
    train_path: str,
    n_estimators: int = 100,
    max_depth: int = 5,
    model_output_path: OutputPath('MLModel') = "models/random_forest.joblib",
):
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    data = np.load(train_path, allow_pickle=True).item()
    X_train, y_train = data["X"], data["y"]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, model_output_path)

@dsl.component(base_image="python:3.10-slim")
def model_evaluation_component(
    model_path: str,
    test_path: str,
    metrics_output_path: OutputPath('JSON') = "artifacts/metrics.json",
):
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)

    data = np.load(test_path, allow_pickle=True).item()
    X_test, y_test = data["X"], data["y"]

    model = joblib.load(model_path)
    preds = model.predict(X_test)

    mse = float(mean_squared_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    metrics = {"mse": mse, "r2": r2}
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    from kfp.v2 import components

    components_dir = os.path.join(os.path.dirname(__file__), "..", "components")
    _ensure_dir(components_dir)

    for func, name in [
        (data_extraction_component, "data_extraction"),
        (data_preprocessing_component, "data_preprocessing"),
        (model_training_component, "model_training"),
        (model_evaluation_component, "model_evaluation"),
    ]:
        yaml_path = os.path.join(components_dir, f"{name}_component.yaml")
        components.OutputPath(yaml_path)
        # compile the component
        components.create_component_from_func(func, output_component_file=yaml_path)
        print(f"Saved component {name} to {yaml_path}")
