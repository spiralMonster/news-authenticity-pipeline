from datetime import datetime

from tfx.components import StatisticsGen
from tfx.components import SchemaGen

def SchemaGeneration(stat_gen:StatisticsGen):
    print(f"[{datetime.now()}] [START] Schema Generator Component.")

    schema_gen=SchemaGen(
        statistics=stat_gen.outputs["statistics"],
        infer_feature_shape=True
    )

    print(f"[{datetime.now()}] [END] Schema Generator Component.")

    return schema_gen