import os
from kfp import dsl
from kfp import Client

from src.pipeline_components import (
    data_extraction_component,
    data_preprocessing_component,
    model_training_component,
    model_evaluation_component,
)


@dsl.pipeline(
    name="boston-housing-mlops-pipeline",
    description="End-to-end pipeline with DVC-backed data, preprocessing, training and evaluation.",
)
def boston_housing_pipeline(
    dvc_remote_url: str = "remote-not-configured",
    raw_csv_path: str = "data/raw/boston_housing.csv",
):
    # Step 1: data extraction
    data_task = data_extraction_component(
        dvc_remote_url=dvc_remote_url,
        output_csv_path=raw_csv_path,
    )

    # Step 2: preprocessing
    prep_task = data_preprocessing_component(
        input_csv_path=data_task.output,
    )

    # Step 3: training
    train_task = model_training_component(
        train_path=prep_task.outputs["output_train_path"],
    )

    # Step 4: evaluation
    eval_task = model_evaluation_component(
        model_path=train_task.output,
        test_path=prep_task.outputs["output_test_path"],
    )


def compile_pipeline(output_path: str = "pipeline.yaml") -> None:
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=boston_housing_pipeline,
        package_path=output_path,
    )
    print(f"Compiled pipeline to {output_path}")


def submit_pipeline(
    host: str,
    experiment_name: str = "boston-housing-experiment",
    run_name: str = "boston-housing-run",
) -> None:
    """Utility function to submit the pipeline to a running KFP cluster."""
    client = Client(host=host)
    experiment = client.create_experiment(name=experiment_name)
    compile_pipeline("pipeline.yaml")
    client.run_pipeline(
        experiment_id=experiment.id,
        job_name=run_name,
        pipeline_package_path="pipeline.yaml",
        params={},
    )


if __name__ == "__main__":
    # For CI and local checks
    compile_pipeline()


