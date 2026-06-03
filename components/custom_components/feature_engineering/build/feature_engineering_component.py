from tfx.components.base.base_component import BaseComponent
from tfx.components.base.executor_spec import ExecutorClassSpec
from tfx.types import channel_utils
from tfx.types.standard_artifacts import Examples
from tfx.types import artifact_utils

from components.custom_components.feature_engineering.build.feature_engineering_component_spec import FeatureEngineeringComponentSpec
from components.custom_components.feature_engineering.build.feature_engineering_executor import FeatureEngineeringExecutor


class FeatureEngineeringComponent(BaseComponent):
    SPEC_CLASS = FeatureEngineeringComponentSpec
    EXECUTOR_SPEC = ExecutorClassSpec(FeatureEngineeringExecutor)

    def __init__(self, examples, feature_engineered_examples=None):
        if not feature_engineered_examples:
            artifact = Examples()
            artifact.split_names = artifact_utils.encode_split_names(["train", "eval", "test"])

            feature_engineered_examples = channel_utils.as_channel([artifact])

        spec = FeatureEngineeringComponentSpec(
            examples=examples,
            feature_engineered_examples=feature_engineered_examples

        )

        super(FeatureEngineeringComponent, self).__init__(spec=spec)

