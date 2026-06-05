from tfx.components.base.base_component import BaseComponent
from tfx.components.base.executor_spec import ExecutorClassSpec
from tfx.types import channel_utils
from tfx.types import artifact_utils
from tfx.types.standard_artifacts import Examples

from components.custom_components.null_data_remover.build.null_data_remover_component_spec import NullDataRemoverComponentSpec
from components.custom_components.null_data_remover.build.null_data_remover_executor import NullDataRemoverExecutor


class NullDataRemover(BaseComponent):
    """
    Custom Component of NullDataRemover
    """

    SPEC_CLASS = NullDataRemoverComponentSpec
    EXECUTOR_SPEC = ExecutorClassSpec(NullDataRemoverExecutor)

    def __init__(self, feature_engineered_examples, preprocessed_examples=None):
        if not preprocessed_examples:
            artifact = Examples()
            artifact.split_names = artifact_utils.encode_split_names(["train", "eval", "test"])

            preprocessed_examples = channel_utils.as_channel([artifact])

        spec = NullDataRemoverComponentSpec(
            feature_engineered_examples=feature_engineered_examples,
            preprocessed_examples=preprocessed_examples
        )

        super(NullDataRemover, self).__init__(spec=spec)
