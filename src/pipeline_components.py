from kfp import dsl
from pipeline_components import (
    data_extraction_component,
    data_preprocessing_component,
    model_training_component,
    model_evaluation_component
)

@dsl.pipeline(
    name="california-housing-mlops-pipeline",
    description="Pipeline for data extraction, preprocessing, training and evaluation."
)
def california_housing_pipeline():

    extract_task = data_extraction_component(
        output_csv_path="data/raw/california.csv"
    )

    preprocess_task = data_preprocessing_component(
        input_csv_path=extract_task.outputs["output_csv_path"]
    )

    train_task = model_training_component(
        train_path=preprocess_task.outputs["output_train_path"]
    )

    eval_task = model_evaluation_component(
        model_path=train_task.outputs["output_model_path"],
        test_path=preprocess_task.outputs["output_test_path"]
    )


if __name__ == "__main__":
    from kfp.v2 import compiler
    compiler.Compiler().compile(
        pipeline_func=california_housing_pipeline,
        package_path="pipeline.yaml"
    )
