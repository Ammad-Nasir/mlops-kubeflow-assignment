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
from kfp.v2.dsl import InputPath, OutputPath


# =========================================================
# Helper
# =========================================================
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# =========================================================
# 1. DATA EXTRACTION
# =========================================================
@dsl.component(base_image="python:3.10-slim")
def data_extraction_component(output_csv_path: OutputPath(str)):
    """Load California housing dataset and write CSV."""
    from sklearn.datasets import fetch_california_housing
    import pandas as pd

    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    df.to_csv(output_csv_path, index=False)


# =========================================================
# 2. DATA PREPROCESSING
# =========================================================
@dsl.component(base_image="python:3.10-slim")
def data_preprocessing_component(
    input_csv_path: InputPath(str),
    output_train_path: OutputPath("NPY"),
    output_test_path: OutputPath("NPY")
):
    """Split data, scale features, save as numpy arrays."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    df = pd.read_csv(input_csv_path)

    X = df.drop("MedHouseVal", axis=1).values
    y = df["MedHouseVal"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    np.save(output_train_path, {"X": X_train_scaled, "y": y_train})
    np.save(output_test_path, {"X": X_test_scaled, "y": y_test})


# =========================================================
# 3. MODEL TRAINING
# =========================================================
@dsl.component(base_image="python:3.10-slim")
def model_training_component(
    train_path: InputPath("NPY"),
    output_model_path: OutputPath("MLModel")
):
    """Train RandomForest model."""
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    import joblib

    data = np.load(train_path, allow_pickle=True).item()
    X_train, y_train = data["X"], data["y"]

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    joblib.dump(model, output_model_path)


# =========================================================
# 4. MODEL EVALUATION
# =========================================================
@dsl.component(base_image="python:3.10-slim")
def model_evaluation_component(
    model_path: InputPath("MLModel"),
    test_path: InputPath("NPY"),
    output_metrics_path: OutputPath("JSON")
):
    """Evaluate model on test set."""
    import numpy as np
    import joblib
    from sklearn.metrics import mean_squared_error, r2_score
    import json

    data = np.load(test_path, allow_pickle=True).item()
    X_test, y_test = data["X"], data["y"]

    model = joblib.load(model_path)
    preds = model.predict(X_test)

    mse = float(mean_squared_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    metrics = {"mse": mse, "r2": r2}

    with open(output_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
