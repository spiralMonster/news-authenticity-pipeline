import os
from pathlib import Path
from datetime import datetime

from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig

from pipelines.init_components import InitComponents
from pipelines.apache_airflow_pipeline.init_pipeline import InitAirflowPipeline


PROJECT_ROOT=Path(__file__).resolve().parents[2]

data_dir=PROJECT_ROOT/"data"
module_dir=PROJECT_ROOT/"modules"
config_dir=PROJECT_ROOT/"configs"

pipeline_root=str(PROJECT_ROOT/"pipeline_run_airflow")
metadata_path=os.path.join(pipeline_root,"metadata.sqlite")


def RunAirflowPipeline(direct_num_workers=1):
    print(f"[START] News Authentication Apache Airflow Pipeline.")

    print(f"[INFO] Creating Beam Pipeline Arguments")
    BEAM_PIPELINE_ARGS=[
        "--runner=DirectRunner",
        f"--direct_number_workers={direct_num_workers}",
        "--direct_running_mode=multi_threading"
    ]

    print(f"[INFO] Creating Airflow Pipeline Config")
    airflow_config={
        "schedule_interval":None,
        "start_date":datetime(2026,6,25)

    }

    airflow_config=AirflowPipelineConfig(airflow_config)

    components=InitComponents(
        data_dir=data_dir,
        module_dir=module_dir,
        config_dir=config_dir
    )

    print(f"[INFO] Initializing Pipeline.")
    pipeline=InitAirflowPipeline(
        components=components,
        pipeline_root=pipeline_root,
        metadata_path=metadata_path,
        beam_pipeline_args=BEAM_PIPELINE_ARGS
    )

    print(f"[INFO] Running pipeline.")
    DAG=AirflowDagRunner(airflow_config).run(pipeline)

    return DAG