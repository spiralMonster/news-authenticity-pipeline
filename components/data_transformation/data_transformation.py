import os

from tfx.components import Transform

from components.custom_components.schema_updater.build.schema_updater_component import SchemaUpdater
from components.custom_components.null_data_remover.build.null_data_remover_component import NullDataRemover


def DataTransformation(example_gen:NullDataRemover,schema_gen:SchemaUpdater):
    transform_gen=Transform(
        examples=example_gen.outputs["preprocessed_examples"],
        schema=schema_gen.outputs["updated_schema"],
        module_file=os.path.abspath("modules/preprocessing_module.py")
    )

    print(f"[INFO] Data Transformed.")

    return transform_gen
