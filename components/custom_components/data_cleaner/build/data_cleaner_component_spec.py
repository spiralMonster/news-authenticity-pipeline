from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter
from tfx.types.standard_artifacts import Examples


class DataCleanerComponentSpec(ComponentSpec):
    """
    Component Spec for Data Cleaning.
    """

    INPUTS = {
        "feature_engineered_examples": ChannelParameter(type=Examples)
    }

    OUTPUTS = {
        "preprocessed_examples": ChannelParameter(type=Examples)
    }

    PARAMETERS = {}