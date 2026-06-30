import os
import argparse
from dotenv import load_dotenv

from pipelines.apache_beam_pipeline.run_pipeline import RunBeamPipeline
from pipelines.apache_airflow_pipeline.run_pipeline import RunAirflowPipeline
from pipelines.kubeflow_pipeline.run_pipeline import RunKubeflowPipeline

from utils.get_final_dataset import GetFinalDataset
from utils.convert_data_to_tfrecords import ConvertDataToTFRecords

load_dotenv()

data_dir="data"

true_news_data_path=os.path.join(data_dir,"True.csv")
fake_news_data_path=os.path.join(data_dir,"Fake.csv")

final_dataset_path=os.path.join(data_dir,"final_dataset.csv")

tfrecord_path=os.path.join(data_dir,"tfrecords","final_dataset.tfrecord")


def RunNewsAuthenticationPipeline():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline_orchestrator",
        type=str,
        required=True

    )

    args=parser.parse_args()
    pipeline_orchestrator=args.pipeline_orchestrator

    print(f"[INFO] Generating data for the pipeline run.")
    GetFinalDataset(
        true_news_data_path=true_news_data_path,
        fake_news_data_path=fake_news_data_path
    )

    ConvertDataToTFRecords(
        csv_data_path=final_dataset_path,
        tfrecord_path=tfrecord_path
    )

    if pipeline_orchestrator=="apache_beam":
        print(f"[INFO] Running the Pipeline with Apache Beam")
        RunBeamPipeline(direct_num_workers=2)

    elif pipeline_orchestrator=="apache_airflow":
        print(f"[INFO] Running the Pipeline with Apache Airflow")
        dag=RunAirflowPipeline(direct_num_workers=2)

    elif pipeline_orchestrator=="kubeflow":
        print(f"[INFO] Running the Pipeline with Kubeflow")
        RunBeamPipeline(direct_num_workers=1)

    else:
        print(f"[Warning] No orchestrator to run the pipeline")





if __name__=="__main__":
    RunNewsAuthenticationPipeline()

