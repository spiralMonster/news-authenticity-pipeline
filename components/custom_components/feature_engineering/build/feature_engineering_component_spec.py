from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ChannelParameter
from tfx.types.standard_artifacts import Examples


class FeatureEngineeringComponentSpec(ComponentSpec):
    """
    Componet Spec for custom Feature Engineering Componenet
    """

    INPUTS={
        "examples":ChannelParameter(type=Examples)
    }

    OUTPUTS={
        "feature_engineered_examples":ChannelParameter(type=Examples)
    }

    PARAMETERS={}