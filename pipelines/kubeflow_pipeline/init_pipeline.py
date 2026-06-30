from tfx.orchestration import metadata,pipeline

from typing_extensions import List


def InitKubeflowPipeline(
        components:List,
        pipeline_root:str,
        beam_pipeline_args:List,
        enable_cache:bool=True,
        pipeline_name:str="news-authentication-pipeline"
):

    p=pipeline.Pipeline(
        pipeline_root=pipeline_root,
        pipeline_name=pipeline_name,
        components=components,
        enable_cache=enable_cache,
        beam_pipeline_args=beam_pipeline_args
    )

    return p