from datetime import datetime
from typing_extensions import List,Tuple

from tfx.components import SchemaGen
from components.custom_components.schema_updater.build.schema_updater_component import SchemaUpdater

updating_params = [
    ("text", 0.75)
]

def UpdateSchema(schema_gen:SchemaGen,updating_params:List[Tuple[str,float]]=updating_params):
    print(f"[{datetime.now()}] [START] Schema Updater Component.")

    schema_updater=SchemaUpdater(
        schema=schema_gen.outputs["schema"],
        updating_params=updating_params
    )

    print(f"[{datetime.now()}] [END] Schema Updater Component.")

    return schema_updater
