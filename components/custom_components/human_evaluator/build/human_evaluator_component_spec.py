from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter
from tfx.types.standard_artifacts import ModelEvaluation
from tfx.types.standard_artifacts import ModelBlessing


class HumanEvaluatorComponentSpec(ComponentSpec):
    """
    Component Spec for custom Human Evaluator Component
    """

    INPUTS={
        "model_evaluation":ChannelParameter(type=ModelEvaluation),
        "model_blessing":ChannelParameter(type=ModelBlessing)
    }

    OUTPUTS={
        "human_evaluator_blessing":ChannelParameter(type=ModelBlessing)
    }

    PARAMETERS={
        "human_evaluator_email_id":ExecutionParameter(type=str),
        "num_retries":ExecutionParameter(type=int),
        "waiting_time":ExecutionParameter(type=int)
    }