from datetime import datetime

from tfx.components import ImportExampleGen
from components.custom_components.feature_engineering.build.feature_engineering_component import FeatureEngineeringComponent


def FeatureEngineering(example_gen:ImportExampleGen):
    print(f"[{datetime.now()}] [START] Feature Engineering Component.")

    feature_gen=FeatureEngineeringComponent(
        examples=example_gen.outputs["examples"]
    )

    print(f"[{datetime.now()}] [END] Feature Engineering Component.")
    return feature_gen