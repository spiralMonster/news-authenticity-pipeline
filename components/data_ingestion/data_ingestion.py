import json
from datetime import datetime

from tfx.components import ImportExampleGen
from tfx.proto import example_gen_pb2

def DataIngestion(tfrecord_dir:str,
                  data_split_ratio_config_path:str):

    print(f"[{datetime.now()}] [START] Data Ingestion Component.")

    print(f"[INFO] Loading the Data Splitting Config.")
    with open(data_split_ratio_config_path,"r") as file:
        config=json.load(file)

    train_ratio=config["train_ratio"]
    eval_ratio=config["eval_ratio"]
    test_ratio=config["test_ratio"]

    print("[INFO] Generating Output Config.")

    output_config=example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(
                    name='train',
                    hash_buckets=int(10*train_ratio)
                ),
                example_gen_pb2.SplitConfig.Split(
                    name='eval',
                    hash_buckets=int(10*eval_ratio)
                ),
                example_gen_pb2.SplitConfig.Split(
                    name='test',
                    hash_buckets=int(10*test_ratio)
                )
            ]
        )
    )


    example_gen=ImportExampleGen(
        input_base=tfrecord_dir,
        output_config=output_config
    )

    print(f"[{datetime.now()}] [END] Data Ingestion Component.")

    return example_gen