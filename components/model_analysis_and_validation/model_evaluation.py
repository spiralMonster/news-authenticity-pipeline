import json

import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.proto import config_pb2


from tfx.components import Evaluator
from tfx.components import Trainer
from tfx.dsl.components.common.resolver import Resolver
from components.custom_components.null_data_remover.build.null_data_remover_component import NullDataRemover

#Load Evaluation Config:
with open("configs/model_configs/model_evaluation_config.json","r") as file:
    evaluation_config=json.load(file)


def ModelEvaluator(
        example_gen:NullDataRemover,
        model_trainer:Trainer,
        model_resolver:Resolver
):
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="label")],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name="BinaryAccuracy"),
                    tfma.MetricConfig(class_name="Precision"),
                    tfma.MetricConfig(class_name="Recall"),
                    tfma.MetricConfig(class_name="AUC")
                ],
                thresholds={
                    "BinaryAccuracy": config_pb2.MetricThreshold(
                        value_threshold=config_pb2.GenericValueThreshold(
                            lower_bound={"value":evaluation_config["BinaryAccuracy"]["lower_bound"]}
                        ),
                        change_threshold=config_pb2.GenericChangeThreshold(
                            direction=config_pb2.MetricDirection.HIGHER_IS_BETTER,
                            absolute={"value":evaluation_config["BinaryAccuracy"]["min_change"]}
                        )
                    ),
                    "Precision": config_pb2.MetricThreshold(
                        value_threshold=config_pb2.GenericValueThreshold(
                            lower_bound={"value":evaluation_config["Precision"]["lower_bound"]}
                        ),
                        change_threshold=config_pb2.GenericChangeThreshold(
                            direction=config_pb2.MetricDirection.HIGHER_IS_BETTER,
                            absolute={"value":evaluation_config["Precision"]["min_change"]}
                        )
                    ),
                    "Recall": config_pb2.MetricThreshold(
                        value_threshold=config_pb2.GenericValueThreshold(
                            lower_bound={"value":evaluation_config["Recall"]["lower_bound"]}
                        ),
                        change_threshold=config_pb2.GenericChangeThreshold(
                            direction=config_pb2.MetricDirection.HIGHER_IS_BETTER,
                            absolute={"value":evaluation_config["Recall"]["min_change"]}
                        )
                    ),
                    "AUC": config_pb2.MetricThreshold(
                        value_threshold=config_pb2.GenericValueThreshold(
                            lower_bound={"value":evaluation_config["AUC"]["lower_bound"]}
                        ),
                        change_threshold=config_pb2.GenericChangeThreshold(
                            direction=config_pb2.MetricDirection.HIGHER_IS_BETTER,
                            absolute={"value":evaluation_config["AUC"]["min_change"]}
                        )
                    )

                }
            )
        ]
    )

    model_evaluator=Evaluator(
        examples=example_gen.outputs["preprocessed_examples"],
        model=model_trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config
    )

    print(f"[INFO] Model Evaluated.")

    return model_evaluator






