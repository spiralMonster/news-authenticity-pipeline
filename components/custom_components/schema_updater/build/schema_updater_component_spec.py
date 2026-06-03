from typing_extensions import List,Tuple

from tfx.types import ComponentSpec
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter
from tfx.types.standard_artifacts import Schema


class SchemaUpdaterComponentSpec(ComponentSpec):
    """
    Component Spec for custom Schema Updater component
    """

    INPUTS={
        "schema":ChannelParameter(type=Schema)
    }

    OUTPUTS={
        "updated_schema":ChannelParameter(type=Schema)
    }

    PARAMETERS={
        "updating_params":ExecutionParameter(type=List[Tuple[str,float]])
    }

