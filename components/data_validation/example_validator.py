from tfx.components import StatisticsGen
from tfx.components import ExampleValidator

from components.custom_components.schema_updater.build.schema_updater_component import SchemaUpdater


def ExampleValidation(stat_gen:StatisticsGen,updated_schema_gen:SchemaUpdater):
    validator=ExampleValidator(
        statistics=stat_gen.outputs["statistics"],
        schema=updated_schema_gen.outputs["updated_schema"]
    )

    print(f"[INFO] Example Validated.")

    return validator