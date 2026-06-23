from datetime import datetime

from tfx.components import Evaluator

from components.custom_components.human_evaluator.build.human_evaluator_component import HumanEvaluator


def HumanEvaluation(
        evaluator:Evaluator,
        human_evaluator_email_id:str,
        num_retries:int,
        waiting_time:int
):
    print(f"[{datetime.now()}] [START] Human Evaluation Component.")

    human_evaluator=HumanEvaluator(
        model_evaluation=evaluator.outputs["evaluation"],
        model_blessing=evaluator.outputs["blessing"],
        human_evaluator_email_id=human_evaluator_email_id,
        num_retries=num_retries,
        waiting_time=waiting_time
    )

    print(f"[{datetime.now()}] [END] Human Evaluation Component.")

    return human_evaluator