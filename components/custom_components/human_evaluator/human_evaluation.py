import json
from datetime import datetime

from tfx.components import Evaluator

from components.custom_components.human_evaluator.build.human_evaluator_component import HumanEvaluator


def HumanEvaluation(
        evaluator:Evaluator,
        config_path:str
):
    print(f"[{datetime.now()}] [START] Human Evaluation Component.")

    print(f"[INFO] Loading Config of Human Evaluator Component.")
    with open(config_path,"r") as file:
        config=json.load(file)

    receiver_email_id=config["receiver_email_id"]
    num_retries=config["num_retries"]
    wait_time=config["wait_time"]

    human_evaluator=HumanEvaluator(
        model_evaluation=evaluator.outputs["evaluation"],
        model_blessing=evaluator.outputs["blessing"],
        human_evaluator_email_id=receiver_email_id,
        num_retries=num_retries,
        waiting_time=wait_time
    )

    print(f"[{datetime.now()}] [END] Human Evaluation Component.")

    return human_evaluator