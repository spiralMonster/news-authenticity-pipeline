from components.custom_components.feature_engineering.build.feature_engineering_component import FeatureEngineeringComponent
from components.custom_components.null_data_remover.build.null_data_remover_component import NullDataRemover


def NullDataRemover(feature_gen:FeatureEngineeringComponent):
    null_data_remover=NullDataRemover(
        feature_engineered_examples=feature_gen.outputs["preprocessed_examples"]
    )

    print(f"[INFO] NUll Data Removed.")

    return null_data_remover