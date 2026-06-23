from datetime import datetime

from tfx.components import ImportExampleGen
from tfx.proto import example_gen_pb2

def DataIngestion(tfrecord_dir:str,
                  train_ratio:float =0.6,
                  eval_ratio:float=0.2,
                  test_ratio:float=0.2):

    print(f"[{datetime.now()}] [START] Data Ingestion Component.")

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