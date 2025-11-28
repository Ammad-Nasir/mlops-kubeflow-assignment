import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from kfp import dsl


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# -------------------------------------------------
# COMPONENTS
# -------------------------------------------------

@dsl.component(base_image="python:3.10-slim")
def data_extraction_component(output_csv_path: str) -> str:
    _ensure_dir(os.path.dirname(output_csv_path))
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df["target"] = boston.target
    df.to_csv(output_csv_path, index=False)
    return output_csv_path


@dsl.component(base_image="python:3.10-slim")
def data_preprocessing_component(
    input_csv_path: str,
    output_train_path: dsl.Output[dsl.Artifact],
    output_test_path: dsl.Output[dsl.Artifact],
):
    df = pd.read_csv(input_csv_path)
    X = df.drop("target", axis=1).values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    np.save(output_train_path.path, {"X": X_train, "y": y_train})
    np.save(output_test_path.path, {"X": X_test, "y": y_test})


@dsl.component(base_image="python:3.10-slim")
def model_training_component(
    train_path: str,
    model_output: dsl.Output[dsl.Model],
):
    data = np.load(train_path, allow_pickle=True).item()
    X_train, y_train = data["X"], data["y"]

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    joblib.dump(model, model_output.path)


@dsl.component(base_image="python:3.10-slim")
def model_evaluation_component(
    model_path: str,
    test_path: str,
    metrics_output: dsl.Output[dsl.Metrics],
):
    data = np.load(test_path, allow_pickle=True).item()
    X_test, y_test = data["X"], data["y"]

    model = joblib.load(model_path)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    metrics = {"mse": mse, "r2": r2}
    with open(metrics_output.path, "w") as f:
        json.dump(metrics, f, indent=2)


# -------------------------------------------------
# YAML GENERATION FOR ALL COMPONENTS
# -------------------------------------------------

if __name__ == "__main__":
    components_dir = os.path.join(os.path.dirname(__file__), "..", "components")
    _ensure_dir(components_dir)

    component_list = [
        (data_extraction_component, "data_extraction_component.yaml"),
        (data_preprocessing_component, "data_preprocessing_component.yaml"),
        (model_training_component, "model_training_component.yaml"),
        (model_evaluation_component, "model_evaluation_component.yaml"),
    ]

    for func, filename in component_list:
        comp = func()
        comp.component_spec.save(os.path.join(components_dir, filename))
        print(f"Saved: {filename}")
