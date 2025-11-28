from kfp import dsl
from kfp.dsl import InputPath, OutputPath

# ---------------------------------------------------------
# 1. DATA EXTRACTION COMPONENT
# ---------------------------------------------------------
@dsl.component
def data_extraction_component(dvc_url: str, output_csv_path: OutputPath(str)):
    import pandas as pd
    import os
    import subprocess

    repo_name = "housing_repo"

    # Clone if not exists
    if not os.path.exists(repo_name):
        subprocess.run(["git", "clone", dvc_url, repo_name], check=True)

    # Pull data with DVC
    subprocess.run(["dvc", "pull"], cwd=repo_name, check=True)

    # Read CSV
    csv_path = os.path.join(repo_name, "data/raw/housing.csv")
    df = pd.read_csv(csv_path)

    df.to_csv(output_csv_path, index=False)


# ---------------------------------------------------------
# 2. DATA PREPROCESSING COMPONENT
# ---------------------------------------------------------
@dsl.component
def data_preprocessing_component(
    input_csv_path: InputPath(str),
    output_train_path: OutputPath(str),
    output_test_path: OutputPath(str)
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_csv_path)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv(output_train_path, index=False)
    test_df.to_csv(output_test_path, index=False)


# ---------------------------------------------------------
# 3. TRAINING COMPONENT
# ---------------------------------------------------------
@dsl.component
def model_training_component(
    train_path: InputPath(str),
    output_model_path: OutputPath(str)
):
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    import joblib

    df = pd.read_csv(train_path)

    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, output_model_path)


# ---------------------------------------------------------
# 4. EVALUATION COMPONENT
# ---------------------------------------------------------
@dsl.component
def model_evaluation_component(
    model_path: InputPath(str),
    test_path: InputPath(str),
    output_metrics_path: OutputPath(str)
):
    import pandas as pd
    import joblib
    from sklearn.metrics import mean_squared_error

    df = pd.read_csv(test_path)

    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    model = joblib.load(model_path)

    preds = model.predict(X)
    mse = mean_squared_error(y, preds)

    with open(output_metrics_path, "w") as f:
        f.write(f"MSE: {mse}")
