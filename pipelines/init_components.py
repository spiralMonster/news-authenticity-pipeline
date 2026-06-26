import os
from dotenv import load_dotenv

from components.data_ingestion.data_ingestion import DataIngestion
from components.custom_components.feature_engineering.feature_engineering import FeatureEngineering
from components.data_validation.statistics_generator import StatisticsGeneration
from components.data_validation.schema_generator import SchemaGeneration
from components.custom_components.schema_updater.update_schema import UpdateSchema
from components.data_validation.example_validator import ExampleValidation
from components.custom_components.data_cleaner.data_cleaner import DataCleanerComponent
from components.data_transformation.data_transformation import DataTransformation
from components.model_trainer.model_trainer import ModelTrainer
from components.model_analysis_and_validation.resolver_node import ResolverNode
from components.model_analysis_and_validation.model_evaluation import ModelEvaluator
from components.custom_components.human_evaluator.human_evaluation import HumanEvaluation
from components.model_analysis_and_validation.model_pusher import ModelPusher


load_dotenv()


def InitComponents(data_dir:str,module_dir:str,config_dir:str):
    print("[INFO] Initializing the components of the pipeline.")

    data_ing=DataIngestion(
        tfrecord_dir=os.path.join(data_dir,"tfrecords"),
        data_split_ratio_config_path=os.path.join(config_dir,"component_configs","data_splitting_ratio.json")
    )

    feature_engineering_gen=FeatureEngineering(
        example_gen=data_ing
    )

    stats_gen=StatisticsGeneration(
        feature_gen=feature_engineering_gen
    )

    schema_gen=SchemaGeneration(
        stat_gen=stats_gen
    )

    schema_updater_gen=UpdateSchema(
        schema_gen=schema_gen,
        updating_params_config_path=os.path.join(config_dir,"component_configs","schema_updating_params_config.json")
    )

    example_val=ExampleValidation(
        stat_gen=stats_gen,
        updated_schema_gen=schema_updater_gen
    )

    data_clean_gen=DataCleanerComponent(
        feature_gen=feature_engineering_gen
    )

    data_transform_gen=DataTransformation(
        example_gen=data_clean_gen,
        schema_gen=schema_updater_gen,
        module_path=os.path.join(module_dir,"preprocessing_module.py")
    )

    model_trainer=ModelTrainer(
        transform_gen=data_transform_gen,
        updated_schema_gen=schema_updater_gen,
        module_file=os.path.join(module_dir,"training_module.py"),
        training_args_path=os.path.join(config_dir,"model_configs","training_configs.json")
    )

    model_resolver=ResolverNode()

    model_evaluator=ModelEvaluator(
        example_gen=data_clean_gen,
        model_trainer=model_trainer,
        model_resolver=model_resolver,
        evaluation_config_path=os.path.join(config_dir,"model_configs","model_evaluation_config.json")
    )

    human_evaluator=HumanEvaluation(
        evaluator=model_evaluator,
        config_path=os.path.join(config_dir,"component_configs","human_evaluation_component_config.json")
    )

    model_pusher=ModelPusher(
        model_trainer=model_trainer,
        human_evaluator=human_evaluator,
        serving_dir=os.environ["MODEL_SERVING_DIR"]
    )

    components=[
        data_ing,
        feature_engineering_gen,
        stats_gen,
        schema_gen,
        schema_updater_gen,
        example_val,
        data_clean_gen,
        data_transform_gen,
        model_trainer,
        model_resolver,
        model_evaluator,
        human_evaluator,
        model_pusher
    ]

    print("[INFO] Components of the pipeline are initialized.")

    return components
