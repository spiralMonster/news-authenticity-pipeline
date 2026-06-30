import os
from pathlib import Path

from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner as kubeflow_dag_runner

from pipelines.kubeflow_pipeline.init_pipeline import InitKubeflowPipeline
from pipelines.init_components import InitComponents
from pipelines.kubeflow_pipeline.patch_pipeline_yaml_with_pvc import PatchPipelineYaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

persistent_volume_claim="kubeflow-pvc"
persistent_volume="kubeflow-pv"

LOCAL_PERSISTENT_VOLUME_MOUNT=PROJECT_ROOT/"kubeflow_persistent_volume"
LOCAL_PERSISTENT_VOLUME_MOUNT=str(LOCAL_PERSISTENT_VOLUME_MOUNT)

KUBERNETES_PERSISTENT_VOLUME_MOUNT="/kubeflow_persistent_volume"


data_dir=os.path.join(LOCAL_PERSISTENT_VOLUME_MOUNT,"data")
module_dir=os.path.join(LOCAL_PERSISTENT_VOLUME_MOUNT,"modules")
config_dir=os.path.join(LOCAL_PERSISTENT_VOLUME_MOUNT,"configs")

pipeline_root=os.path.join(LOCAL_PERSISTENT_VOLUME_MOUNT,"pipeline_run_kubeflow")

serving_dir=os.path.join(LOCAL_PERSISTENT_VOLUME_MOUNT,"serving_model_dir")

output_file=os.path.join("pipelines","kubeflow_pipeline","news_authentication_pipeline.yaml")

tfx_image="docker.io/spiralmonster/tfx:news_authentication_pipeline"


def RunKubeflowPipeline(direct_num_workers:int=0):
    print(f"[START] News Authentication Pipeline using Kubeflow.")

    print(f"[INFO] Initializing Components.")
    components=InitComponents(
        data_dir=data_dir,
        module_dir=module_dir,
        config_dir=config_dir,
        serving_dir=serving_dir,
        is_pipeline_orchestrator_kubeflow=True
    )

    print(f"[INFO] Creating Beam Arguments.")
    beam_pipeline_args=[
        f"--direct_number_workers={direct_num_workers}"
    ]

    print(f"[INFO] Initializing Pipeline.")
    pipeline=InitKubeflowPipeline(
        components=components,
        pipeline_root=pipeline_root,
        beam_pipeline_args=beam_pipeline_args
    )

    print(f"[INFO] Creating Runner Config.")
    runner_config=kubeflow_dag_runner.KubeflowV2DagRunnerConfig(
        default_image=tfx_image
    )


    print(f"[INFO] Running Kubeflow Dag Runner.")
    kubeflow_dag_runner.KubeflowV2DagRunner(
        config=runner_config,
        output_filename=output_file
    ).run(pipeline)

    print("[INFO] Patching the Pipelines's Argo configuration for Persistent Volume.")
    PatchPipelineYaml(
        yaml_path=output_file,
        pvc_name=persistent_volume_claim,
        local_mount_path=LOCAL_PERSISTENT_VOLUME_MOUNT,
        kubernetes_mount_path=KUBERNETES_PERSISTENT_VOLUME_MOUNT
    )

    print(f"[INFO] Generated Pipeline's Argo Configuration.")





