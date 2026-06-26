import json
from datetime import datetime

import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.proto import config_pb2


from tfx.components import Evaluator
from tfx.components import Trainer
from tfx.dsl.components.common.resolver import Resolver
from components.custom_components.data_cleaner.build.data_cleaner_component import DataCleaner



def ModelEvaluator(
        example_gen:DataCleaner,
        model_trainer:Trainer,
        model_resolver:Resolver,
        evaluation_config_path:str
):
    print(f"[{datetime.now()}] [START] Model Evaluator Component.")

    print(f"[INFO] Loading Evaluation Config.")
    with open(evaluation_config_path,"r") as file:
        evaluation_config=json.load(file)


    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(label_key="label")
        ],
        slicing_specs=[
            tfma.SlicingSpec()
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(
                        class_name="BinaryAccuracy",
                        threshold=config_pb2.MetricThreshold(
                            value_threshold=config_pb2.GenericValueThreshold(
                                lower_bound={"value":evaluation_config["BinaryAccuracy"]["lower_bound"]}
                            ),
                            change_threshold=config_pb2.GenericChangeThreshold(
                                direction=config_pb2.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value":evaluation_config["BinaryAccuracy"]["min_change"]}
                            )
                        )
                    ),

                    tfma.MetricConfig(
                        class_name="Precision",
                        threshold=config_pb2.MetricThreshold(
                            value_threshold=config_pb2.GenericValueThreshold(
                                lower_bound={"value":evaluation_config["Precision"]["lower_bound"]}
                            ),
                            change_threshold=config_pb2.GenericChangeThreshold(
                                direction=config_pb2.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value":evaluation_config["Precision"]["min_change"]}
                            )
                        )
                    ),

                    tfma.MetricConfig(
                        class_name="Recall",
                        threshold=config_pb2.MetricThreshold(
                            value_threshold=config_pb2.GenericValueThreshold(
                                lower_bound={"value":evaluation_config["Recall"]["lower_bound"]}
                            ),
                            change_threshold=config_pb2.GenericChangeThreshold(
                                direction=config_pb2.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value":evaluation_config["Recall"]["min_change"]}
                            )
                        )
                    ),

                    tfma.MetricConfig(
                        class_name="AUC",
                        threshold=config_pb2.MetricThreshold(
                            value_threshold=config_pb2.GenericValueThreshold(
                                lower_bound={"value":evaluation_config["AUC"]["lower_bound"]}
                            ),
                            change_threshold=config_pb2.GenericChangeThreshold(
                                direction=config_pb2.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value":evaluation_config["AUC"]["min_change"]}
                            )
                        )
                    ),
                ]
            )
        ]
    )

    model_evaluator=Evaluator(
        examples=example_gen.outputs["preprocessed_examples"],
        model=model_trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
        example_splits=["test"]
    )

    print(f"[{datetime.now()}] [END] Model Evaluator Component.")

    return model_evaluator






