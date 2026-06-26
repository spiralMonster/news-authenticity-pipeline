from tfx.orchestration import metadata,pipeline

from typing_extensions import List


def InitAirflowPipeline(
        components:List,
        pipeline_root:str,
        metadata_path:str,
        beam_pipeline_args:List,
        enable_cache:bool=True,
        pipeline_name:str="news_authentication_airflow_pipeline"
):


    metadata_config=metadata.sqlite_metadata_connection_config(
        metadata_path
    )

    p=pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=enable_cache,
        metadata_connection_config=metadata_config,
        beam_pipeline_args=beam_pipeline_args
    )

    return p

