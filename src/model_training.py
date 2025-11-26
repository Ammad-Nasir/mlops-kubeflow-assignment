import os
import json
from typing import Dict

import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def train_and_evaluate(
    train_path: str,
    test_path: str,
    model_output_path: str = "models/random_forest.joblib",
    metrics_output_path: str = "artifacts/metrics.json",
    n_estimators: int = 100,
    max_depth: int = 5,
) -> Dict[str, float]:
    """Local utility to train and evaluate the model without Kubeflow."""
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)

    train = np.load(train_path, allow_pickle=True).item()
    test = np.load(test_path, allow_pickle=True).item()

    X_train, y_train = train["X"], train["y"]
    X_test, y_test = test["X"], test["y"]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = float(mean_squared_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    joblib.dump(model, model_output_path)
    metrics = {"mse": mse, "r2": r2}
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    # Example usage for quick local debugging
    metrics = train_and_evaluate(
        train_path="data/processed/train.npy",
        test_path="data/processed/test.npy",
    )
    print("Metrics:", metrics)


