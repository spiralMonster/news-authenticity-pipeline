from tfx.components.base.base_component import BaseComponent
from tfx.components.base.executor_spec import ExecutorClassSpec
from tfx.types.standard_artifacts import Schema
from tfx.types import channel_utils

from components.custom_components.schema_updater.build.schema_updater_component_spec import SchemaUpdaterComponentSpec
from components.custom_components.schema_updater.build.schema_updater_executor import SchemaUpdaterExecutor

class SchemaUpdater(BaseComponent):
    SPEC_CLASS=SchemaUpdaterComponentSpec
    EXECUTOR_SPEC=ExecutorClassSpec(SchemaUpdaterExecutor)

    def __init__(self,schema,updating_params,updated_schema=None):

        if not updated_schema:
            artifact=Schema()

            updated_schema=channel_utils.as_channel([artifact])

        spec=SchemaUpdaterComponentSpec(
            schema=schema,
            updated_schema=updated_schema,
            updating_params=updating_params
        )

        super(SchemaUpdater,self).__init__(spec=spec)