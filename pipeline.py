from kfp import dsl, compiler
from src.pipeline_components import (
    data_extraction_component,
    data_preprocessing_component,
    model_training_component,
    model_evaluation_component,
)


@dsl.pipeline(
    name="boston-housing-mlops-pipeline",
    description="End-to-end MLOps pipeline using DVC, preprocessing, training, and evaluation."
)
def boston_housing_pipeline(
    dvc_url: str = "https://github.com/user/housing-dvc.git"
):

    # 1. Extract dataset
    extract_task = data_extraction_component(
        dvc_url=dvc_url
    )

    # 2. Preprocess
    prep_task = data_preprocessing_component(
        input_csv_path=extract_task.output
    )

    # 3. Train
    train_task = model_training_component(
        train_path=prep_task.outputs["output_train_path"]
    )

    # 4. Evaluate
    eval_task = model_evaluation_component(
        model_path=train_task.output,
        test_path=prep_task.outputs["output_test_path"]
    )


# -------------------------
# Compile pipeline
# -------------------------
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=boston_housing_pipeline,
        package_path="pipeline.yaml",
    )
