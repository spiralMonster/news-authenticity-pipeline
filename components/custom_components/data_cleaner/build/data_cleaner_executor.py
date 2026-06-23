import os
import json
import pandas as pd
import tensorflow as tf

from typing_extensions import Text,Dict,List,Any

from tfx.components.base.base_executor import BaseExecutor
from tfx.types.standard_artifacts import Artifact
from tfx.types import artifact_utils


class DataCleanerExecutor(BaseExecutor):
    """
    Executor for custom Data Cleaner Component
    """

    def convert_tfrecord_to_datframe(self, tfrecord_dir: str):
        feature_description = {
            "num_single_quote_error": tf.io.FixedLenFeature([], tf.int64),
            "num_spacing_error": tf.io.FixedLenFeature([], tf.int64),
            "num_space_absence_after_sentence_completion": tf.io.FixedLenFeature([], tf.int64),
            "num_capitalized_words": tf.io.FixedLenFeature([], tf.int64),
            "num_capitalization_absence_after_sentence_completion": tf.io.FixedLenFeature([], tf.int64),
            "num_spelling_errors": tf.io.FixedLenFeature([], tf.int64),
            "num_punctuations": tf.io.FixedLenFeature([], tf.int64),
            "num_numeric_values": tf.io.FixedLenFeature([], tf.int64),
            "text": tf.io.FixedLenFeature([], tf.string, default_value=''),
            "num_words": tf.io.FixedLenFeature([], tf.int64, default_value=0),
            "label": tf.io.FixedLenFeature([], tf.int64)
        }

        def parse_example(example_proto):
            result = tf.io.parse_single_example(
                example_proto,
                feature_description
            )

            return result

        tf_record_file = os.path.join(tfrecord_dir, os.listdir(tfrecord_dir)[0])

        tfrecord_dataset = tf.data.TFRecordDataset(
            tf_record_file,
            compression_type="GZIP"
        )

        tfrecord_dataset = tfrecord_dataset.map(parse_example)

        rows = []

        for item in tfrecord_dataset:
            row = {
                "num_single_quote_error": item["num_single_quote_error"].numpy(),
                "num_spacing_error": item["num_spacing_error"].numpy(),
                "num_space_absence_after_sentence_completion": item[
                    "num_space_absence_after_sentence_completion"].numpy(),
                "num_capitalized_words": item["num_capitalized_words"].numpy(),
                "num_capitalization_absence_after_sentence_completion": item[
                    "num_capitalization_absence_after_sentence_completion"].numpy(),
                "num_spelling_errors": item["num_spelling_errors"].numpy(),
                "num_punctuations": item["num_punctuations"].numpy(),
                "num_numeric_values": item["num_numeric_values"].numpy(),
                "text": item["text"].numpy().decode(),
                "num_words": item["num_words"].numpy(),
                "label": item["label"].numpy()
            }

            rows.append(row)

        dataset = pd.DataFrame(rows)

        return dataset

    def remove_null_values_from_dataset(self, dataset: pd.DataFrame):
        dataset["text"] = dataset["text"].map(lambda text: text if text != '' else None)

        dataset = dataset.dropna(inplace=False)

        return dataset

    def dataframe_to_tfrecord(self, data: pd.DataFrame, tfrecord_dir: str):

        def _bytes_feature(value):
            feat = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[value])
            )

            return feat

        def _int64_feature(value):
            feat = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[value])
            )

            return feat

        tfrecord_path = os.path.join(tfrecord_dir, "tf_record.gz")

        tf_record_option = tf.io.TFRecordOptions(compression_type="GZIP")
        writer = tf.io.TFRecordWriter(
            tfrecord_path,
            options=tf_record_option
        )

        for _, row in data.iterrows():
            feature = {
                "num_single_quote_error": _int64_feature(value=row["num_single_quote_error"]),
                "num_spacing_error": _int64_feature(value=row["num_spacing_error"]),
                "num_space_absence_after_sentence_completion": _int64_feature(
                    value=row["num_space_absence_after_sentence_completion"]),
                "num_capitalized_words": _int64_feature(value=row["num_capitalized_words"]),
                "num_capitalization_absence_after_sentence_completion": _int64_feature(
                    value=row["num_capitalization_absence_after_sentence_completion"]),
                "num_spelling_errors": _int64_feature(value=row["num_spelling_errors"]),
                "num_punctuations": _int64_feature(value=row["num_punctuations"]),
                "num_numeric_values": _int64_feature(value=row["num_numeric_values"]),
                "label": _int64_feature(value=row["label"])

            }

            if row["text"] != '':
                feature["text"] = _bytes_feature(value=row["text"].encode("utf-8"))

            if row["num_words"] != 0:
                feature["num_words"] = _int64_feature(value=row["num_words"])

            example = tf.train.Example(
                features=tf.train.Features(feature=feature)
            )

            writer.write(example.SerializeToString())

        writer.close()

    def Do(
            self,
            input_dict: Dict[Text, List[Artifact]],
            output_dict: Dict[Text, List[Artifact]],
            exec_properties: Dict[Text, Any]
    ):
        self._log_startup(input_dict, output_dict, exec_properties)

        inp_train_dir = artifact_utils.get_split_uri(input_dict["feature_engineered_examples"], "train")
        inp_val_dir = artifact_utils.get_split_uri(input_dict["feature_engineered_examples"], "eval")
        inp_test_dir = artifact_utils.get_split_uri(input_dict["feature_engineered_examples"], "test")

        output_artifact = output_dict["preprocessed_examples"][-1]
        output_artifact.split_names = artifact_utils.encode_split_names(["train", "eval", "test"])

        out_train_dir = artifact_utils.get_split_uri(output_dict["preprocessed_examples"], "train")
        out_val_dir = artifact_utils.get_split_uri(output_dict["preprocessed_examples"], "eval")
        out_test_dir = artifact_utils.get_split_uri(output_dict["preprocessed_examples"], "test")

        input_dirs = [inp_train_dir, inp_val_dir, inp_test_dir]
        output_dirs = [out_train_dir, out_val_dir, out_test_dir]

        number_of_examples={}

        for (inp_dir, out_dir) in zip(input_dirs, output_dirs):
            dataset = self.convert_tfrecord_to_datframe(tfrecord_dir=inp_dir)
            dataset = self.remove_null_values_from_dataset(dataset=dataset)

            split_name=inp_dir.split("/")[-1]
            number_of_examples[f"number_of_examples_in_{split_name}_data"]=len(dataset)

            tf.io.gfile.makedirs(out_dir)
            self.dataframe_to_tfrecord(data=dataset, tfrecord_dir=out_dir)


        with open("configs/model_configs/number_of_examples.json","w") as file:
            json.dump(number_of_examples,file)

        print("[INFO] Number of examples in each split logged")





