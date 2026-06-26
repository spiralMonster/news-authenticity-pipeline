from datetime import datetime

from components.custom_components.feature_engineering.build.feature_engineering_component import FeatureEngineeringComponent
from components.custom_components.data_cleaner.build.data_cleaner_component import DataCleaner

def DataCleanerComponent(feature_gen:FeatureEngineeringComponent):
    print(f"[{datetime.now()}] [START] Data Cleaner Component.")

    data_cleaner=DataCleaner(
        feature_engineered_examples=feature_gen.outputs["feature_engineered_examples"]
    )

    print(f"[{datetime.now()}] [END] Data Cleaner Component.")

    return data_cleaner