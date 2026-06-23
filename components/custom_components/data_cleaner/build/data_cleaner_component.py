from tfx.components.base.base_component import BaseComponent
from tfx.components.base.executor_spec import ExecutorClassSpec
from tfx.types import channel_utils
from tfx.types import artifact_utils
from tfx.types.standard_artifacts import Examples

from components.custom_components.data_cleaner.build.data_cleaner_component_spec import DataCleanerComponentSpec
from components.custom_components.data_cleaner.build.data_cleaner_executor import DataCleanerExecutor


class DataCleaner(BaseComponent):
    """
    Custom Component of Data Cleaner
    """

    SPEC_CLASS = DataCleanerComponentSpec
    EXECUTOR_SPEC = ExecutorClassSpec(DataCleanerExecutor)

    def __init__(self, feature_engineered_examples, preprocessed_examples=None):
        if not preprocessed_examples:
            artifact = Examples()
            artifact.split_names = artifact_utils.encode_split_names(["train", "eval", "test"])

            preprocessed_examples = channel_utils.as_channel([artifact])

        spec = DataCleanerComponentSpec(
            feature_engineered_examples=feature_engineered_examples,
            preprocessed_examples=preprocessed_examples
        )

        super(DataCleaner, self).__init__(spec=spec)
