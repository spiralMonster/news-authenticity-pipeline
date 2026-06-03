from tfx.components import ImportExampleGen
from components.custom_components.feature_engineering.build.feature_engineering_component import FeatureEngineeringComponent


def FeatureEngineering(example_gen:ImportExampleGen):
    feature_gen=FeatureEngineeringComponent(
        examples=example_gen.outputs["examples"]
    )

    print(f"[INFO] Feature Engineering Completed.")
    return feature_gen