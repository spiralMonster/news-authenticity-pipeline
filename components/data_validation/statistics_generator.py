from datetime import datetime

from tfx.components import StatisticsGen
from components.custom_components.feature_engineering.build.feature_engineering_component import FeatureEngineeringComponent

def StatisticsGeneration(feature_gen:FeatureEngineeringComponent):
    print(f"[{datetime.now()}] [START] Statistics Generator Component.")

    stat_gen=StatisticsGen(
        examples=feature_gen.outputs["feature_engineered_examples"]
    )

    print(f"[{datetime.now()}] [END] Statistics Generator Component.")

    return stat_gen