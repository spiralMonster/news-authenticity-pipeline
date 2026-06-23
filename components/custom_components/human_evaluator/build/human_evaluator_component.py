from tfx.components.base.base_component import BaseComponent
from tfx.components.base.executor_spec import ExecutorClassSpec
from tfx.types import channel_utils
from tfx.types.standard_artifacts import ModelBlessing

from components.custom_components.human_evaluator.build.human_evaluator_component_spec import HumanEvaluatorComponentSpec
from components.custom_components.human_evaluator.build.human_evaluator_executor import HumanEvaluatorExecutor


class HumanEvaluator(BaseComponent):
    """
    The Custom Component of Human Evaluation by sending the details of the pipeline run to the Human Evaluator.
    """

    SPEC_CLASS = HumanEvaluatorComponentSpec
    EXECUTOR_SPEC = ExecutorClassSpec(HumanEvaluatorExecutor)

    def __init__(
            self,
            model_evaluation,
            model_blessing,
            human_evaluator_email_id,
            num_retries,
            waiting_time,
            human_evaluator_blessing=None

    ):
        """
        Args:
        :param model_evaluation: The evaluation done by Evaluator Component.
        :param model_blessing: The blessing provided by Evaluator Component.
        :param human_evaluator_email_id: The email id of human evaluator.
        :param num_retries: Number of retries if there is some issue with email sending/receiving.
        :param waiting_time: Wait time (in seconds) before the reply of human evaluator is read.
        :param human_evaluator_blessing: The blessing of human evaluator
        """

        if not human_evaluator_blessing:
            artifact = ModelBlessing()

            human_evaluator_blessing = channel_utils.as_channel([artifact])

        spec = HumanEvaluatorComponentSpec(
            model_evaluation=model_evaluation,
            model_blessing=model_blessing,
            human_evaluator_email_id=human_evaluator_email_id,
            num_retries=num_retries,
            waiting_time=waiting_time,
            human_evaluator_blessing=human_evaluator_blessing
        )

        super(HumanEvaluator, self).__init__(spec=spec)
