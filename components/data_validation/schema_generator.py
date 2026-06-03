from tfx.components import StatisticsGen
from tfx.components import SchemaGen

def SchemaGeneration(stat_gen:StatisticsGen):
    schema_gen=SchemaGen(
        statistics=stat_gen.outputs["statistics"],
        infer_feature_shape=True
    )

    print(f"[INFO] Schema Generated.")

    return schema_gen