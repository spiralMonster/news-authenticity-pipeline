from tfx.components import StatisticsGen
from components.custom_components.feature_engineering.build.feature_engineering_component import FeatureEngineeringComponent

def StatisticsGeneration(feature_gen:FeatureEngineeringComponent):
    stat_gen=StatisticsGen(
        examples=feature_gen.outputs["feature_engineered_examples"]
    )

    print(f"[INFO] Statistics Generated.")

    return stat_gen