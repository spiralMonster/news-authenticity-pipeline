import os
from datetime import datetime
import json

from tfx.components import Trainer
from tfx.components import Transform

from tfx.components.base.executor_spec import ExecutorClassSpec
from tfx.components.trainer.executor import GenericExecutor
from tfx.proto import trainer_pb2

from components.custom_components.schema_updater.update_schema import SchemaUpdater



def ModelTrainer(transform_gen:Transform,
                 updated_schema_gen:SchemaUpdater,
                 module_file:str,
                 training_args_path:str
                 ):

    print(f"[{datetime.now()}] [START] Model Trainer Component.")

    print(f"[INFO] Loading Training Args.")
    with open(training_args_path,"r") as file:
        training_args=json.load(file)

    trainer=Trainer(
        module_file=os.path.abspath(module_file),
        custom_executor_spec=ExecutorClassSpec(GenericExecutor),
        transformed_examples=transform_gen.outputs["transformed_examples"],
        transform_graph=transform_gen.outputs["transform_graph"],
        schema=updated_schema_gen.outputs["updated_schema"],
        train_args=trainer_pb2.TrainArgs(num_steps=training_args["NUM_TRAIN_STEPS"]),
        eval_args=trainer_pb2.EvalArgs(num_steps=training_args["NUM_EVAL_STEPS"])
    )

    print(f"[{datetime.now()}] [END] Model Trainer Component.")

    return trainer
