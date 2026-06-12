import os
from dotenv import load_dotenv
from pyarrow import filesystem

from tfx.proto import pusher_pb2

from tfx.components import Pusher
from tfx.components import Trainer
from tfx.components import Evaluator


def ModelPusher(
        model_trainer:Trainer,
        model_evaluator:Evaluator
):
    serving_dir=os.environ["MODEL_SERVING_DIR"]

    model_pusher=Pusher(
        model=model_trainer.outputs["model"],
        model_blessing=model_evaluator.outputs["model_blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_dir
            )
        )
    )

    print(f"[INFO] Model has been pushed.")

    return model_pusher