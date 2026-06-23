import os
from datetime import datetime

from tfx.proto import pusher_pb2

from tfx.components import Pusher
from tfx.components import Trainer

from components.custom_components.human_evaluator.build.human_evaluator_component import HumanEvaluator


def ModelPusher(
        model_trainer:Trainer,
        human_evaluator:HumanEvaluator
):
    print(f"[{datetime.now()}] [START] Model Pusher Component.")
    serving_dir=os.environ["MODEL_SERVING_DIR"]

    model_pusher=Pusher(
        model=model_trainer.outputs["model"],
        model_blessing=human_evaluator.outputs["human_evaluator_blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_dir
            )
        )
    )

    print(f"[{datetime.now()}] [END] Model Pusher Component.")

    return model_pusher