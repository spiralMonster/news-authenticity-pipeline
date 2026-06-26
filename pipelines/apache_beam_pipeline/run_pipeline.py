import os
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

from pipelines.init_components import InitComponents
from pipelines.apache_beam_pipeline.init_pipeline import InitBeamPipeline


data_dir="data"
module_dir="modules"
config_dir="configs"

pipeline_root="pipeline_run_beam"
requirements_file="requirements.txt"
metadata_path=os.path.join(pipeline_root,"metadata.sqlite")


def RunBeamPipeline(direct_num_workers:int=1):
    print(f"[START] News Authentication Apache Beam Pipeline.")

    components=InitComponents(
        data_dir=data_dir,
        module_dir=module_dir,
        config_dir=config_dir
    )

    print(f"[INFO] Creating Beam Pipeline Arguments")
    BEAM_PIPELINE_ARGS = [
        "--runner=DirectRunner",
        f"--direct_num_workers={direct_num_workers}",
        "--direct_running_mode=multi_threading",
    ]


    print(f"[INFO] Initializing Pipeline.")
    pipeline=InitBeamPipeline(
        components=components,
        pipeline_root=pipeline_root,
        requirement_file=requirements_file,
        metadata_path=metadata_path,
        beam_pipeline_args=BEAM_PIPELINE_ARGS
    )

    print(f"[INFO] Running Pipeline")

    BeamDagRunner().run(pipeline)

    print(f"[END] Pipeline Run completed successfully.")