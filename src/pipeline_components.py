import os
import json
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from kfp import dsl


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_and_save_boston_csv(output_path: str) -> str:
    """Utility: download/load Boston housing and save as CSV for DVC tracking."""
    _ensure_dir(os.path.dirname(output_path))
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df["target"] = boston.target
    df.to_csv(output_path, index=False)
    return output_path


@dsl.component(base_image="python:3.10-slim")
def data_extraction_component(dvc_remote_url: str, output_csv_path: str) -> str:
    """Fetch versioned dataset using DVC (expected to be configured in the image).

    In practice, this would run `dvc pull` or `dvc get`. Here we assume the data
    is already present in the container image or mounted volume.
    """
    import os

    if not os.path.exists(output_csv_path):
        raise FileNotFoundError(
            f"Expected dataset at {output_csv_path}. "
            "In your environment, fetch it with `dvc get` or `dvc pull` before running."
        )
    return output_csv_path


@dsl.component(base_image="python:3.10-slim")
def data_preprocessing_component(
    input_csv_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    output_train_path: str = "data/processed/train.npy",
    output_test_path: str = "data/processed/test.npy",
) -> Tuple[str, str]:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import os

    os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_test_path), exist_ok=True)

    df = pd.read_csv(input_csv_path)
    X = df.drop("target", axis=1).values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    np.save(output_train_path, {"X": X_train_scaled, "y": y_train})
    np.save(output_test_path, {"X": X_test_scaled, "y": y_test})

    return output_train_path, output_test_path


@dsl.component(base_image="python:3.10-slim")
def model_training_component(
    train_path: str,
    n_estimators: int = 100,
    max_depth: int = 5,
    model_output_path: str = "models/random_forest.joblib",
) -> str:
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    import os

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
    return model_output_path


@dsl.component(base_image="python:3.10-slim")
def model_evaluation_component(
    model_path: str,
    test_path: str,
    metrics_output_path: str = "artifacts/metrics.json",
) -> str:
    import numpy as np
    import joblib
    import json
    from sklearn.metrics import mean_squared_error, r2_score
    import os

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

    return metrics_output_path


if __name__ == "__main__":
    # Helper script to compile components to YAML in ../components
    from kfp import components

    components_dir = os.path.join(os.path.dirname(__file__), "..", "components")
    _ensure_dir(components_dir)

    for func, name in [
        (data_extraction_component, "data_extraction"),
        (data_preprocessing_component, "data_preprocessing"),
        (model_training_component, "model_training"),
        (model_evaluation_component, "model_evaluation"),
    ]:
        comp = components.create_component_from_func(func)
        yaml_path = os.path.join(components_dir, f"{name}_component.yaml")
        comp.save(yaml_path)
        print(f"Saved component {name} to {yaml_path}")


