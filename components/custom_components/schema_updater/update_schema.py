import json
from datetime import datetime

from tfx.components import SchemaGen
from components.custom_components.schema_updater.build.schema_updater_component import SchemaUpdater

def UpdateSchema(schema_gen:SchemaGen,updating_params_config_path:str):
    print(f"[{datetime.now()}] [START] Schema Updater Component.")

    print(f"[INFO] Loading Updating Parameters Config.")
    with open(updating_params_config_path,"r") as file:
        config=json.load(file)

    updating_params=[]
    for key,value in config.items():
        updt=(key,value)
        updating_params.append(updt)

    schema_updater=SchemaUpdater(
        schema=schema_gen.outputs["schema"],
        updating_params=updating_params
    )

    print(f"[{datetime.now()}] [END] Schema Updater Component.")

    return schema_updater
