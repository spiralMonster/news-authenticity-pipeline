from datetime import datetime
import os

from tfx.components import Transform

from components.custom_components.schema_updater.build.schema_updater_component import SchemaUpdater
from components.custom_components.data_cleaner.build.data_cleaner_component import DataCleaner


def DataTransformation(example_gen:DataCleaner,schema_gen:SchemaUpdater,module_path:str):
    print(f"[{datetime.now()}] [START] Data Transformation Component.")

    transform_gen=Transform(
        examples=example_gen.outputs["preprocessed_examples"],
        schema=schema_gen.outputs["updated_schema"],
        module_file=os.path.abspath(module_path)
    )

    print(f"[{datetime.now()}] [END] Data Transformation Component.")

    return transform_gen
