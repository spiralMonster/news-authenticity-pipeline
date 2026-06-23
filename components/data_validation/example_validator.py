from datetime import datetime

from tfx.components import StatisticsGen
from tfx.components import ExampleValidator

from components.custom_components.schema_updater.build.schema_updater_component import SchemaUpdater


def ExampleValidation(stat_gen:StatisticsGen,updated_schema_gen:SchemaUpdater):
    print(f"[{datetime.now()}] [START] Data Validation Component.")

    validator=ExampleValidator(
        statistics=stat_gen.outputs["statistics"],
        schema=updated_schema_gen.outputs["updated_schema"]
    )

    print(f"[{datetime.now()}] [END] Data Validation Component.")

    return validator