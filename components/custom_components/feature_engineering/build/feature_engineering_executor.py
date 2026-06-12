import os
import tensorflow as tf
import pandas as pd
import string
from spellchecker import SpellChecker
from typing_extensions import Text,Dict,List,Any

from tfx.components.base.base_executor import BaseExecutor
from tfx.types import Artifact
from tfx.types import artifact_utils


class FeatureEngineeringExecutor(BaseExecutor):
    """
    Executor for custom Feature Engineering Component
    """

    def convert_tfrecord_to_dataframe(self, tfrecord_file: str):
        feature_description = {
            "text": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.string)
        }

        def parse_example(example_proto):
            result = tf.io.parse_single_example(
                example_proto,
                feature_description
            )

            return result

        dataset = tf.data.TFRecordDataset(
            tfrecord_file,
            compression_type="GZIP"
        )
        dataset = dataset.map(parse_example)

        rows = []

        for item in dataset:
            row = {
                "text": item["text"].numpy().decode(),
                "label": item["label"].numpy().decode()
            }

            rows.append(row)

        dataframe = pd.DataFrame(rows)

        return dataframe

    def generate_features(
            self,
            data: pd.DataFrame
    ):
        """
        Generate the features from the 'text' field of data.
        The functions to generate various features are:

          - num_single_quote_error: Counts the number of single quote error in text.

          - num_spacing_error: Counts the number of extra spaces between two words.

          - num_space_absence_after_sentence_completion: Counts the number of instances where there is no
            space after sentence completion.

          - num_capitalized_words: Counts the number of capitalized word in text.

          - num_capitalization_absence_after_sentence_completion: Counts the number of instances where the
            first letter of the word after the sentence completion is not capitalized.

          - num_spelling_errors: Counts the number of mispelled words in the text.

          - num_words: Counts the number of words in text.

          - num_punctuations: Counts the number of punctuation marks in the text.

          - num_numeric_values: counts the number of numeric value in text.


        """

        def num_single_quote_error(text: str):
            num_errors = 0
            words = text.split(" ")

            ind1 = 0
            while ind1 < len(words) - 1:
                ind2 = ind1 + 1

                word1 = words[ind1]
                word2 = words[ind2]

                if word1.isalpha() and word2.isalpha():
                    word1 = word1.lower()
                    word2 = word2.lower()

                    if len(word1) > 1 and len(word2) == 1:
                        if word2 not in ['a', 'i']:
                            num_errors += 1

                    elif word1 == 'i' and (word2 == "ll" or word2 == "d"):
                        num_errors += 1

                ind1 += 1

            return num_errors

        def num_spacing_error(text: str):
            words = text.split(" ")
            num_errors = words.count("")

            return num_errors

        def num_space_absence_after_sentence_completion(text: str):
            num_errors = 0
            words = text.split(" ")

            for word in words:
                if any(punct in word for punct in ['.', '?', '!']):
                    if word[-1] not in ['.', '?', '!']:
                        num_errors += 1

            return num_errors

        def num_capitalized_words(text: str):
            num_cap_words = 0
            words = text.split(" ")

            table = str.maketrans('', '', string.punctuation)

            for word in words:
                word = word.translate(table)
                if word.isalpha():
                    if word.isupper():
                        if word != "I":
                            num_cap_words += 1

            return num_cap_words

        def num_capitalization_absence_after_sentence_completion(text: str):
            num_errors = 0
            punct_to_check_for = ['.', '?', '!']

            for punct in punct_to_check_for:
                sent = text.split(punct)
                sent = [s.strip() for s in sent if s != '']

                for s in sent:
                    if s != '':
                        first_letter = s[0]
                        if first_letter.isalpha():
                            if not first_letter.isupper():
                                num_errors += 1

            return num_errors

        spell = SpellChecker()

        def num_spelling_errors(text: str):
            num_errors = 0
            table = str.maketrans('', '', string.punctuation)

            words = text.split(" ")
            for word in words:
                word = word.translate(table)
                if word.isalpha():
                    word = word.lower()
                    if word not in spell:
                        num_errors += 1

            return num_errors

        def num_words(text: str):
            words = text.split(" ")
            num = len(words) - words.count("")
            return num

        punct = list(string.punctuation)

        def num_punctuations(text: str):
            num = 0
            words = text.split(" ")

            for word in words:
                if any(p in word for p in punct):
                    num += 1

            return num

        numeric_values = list("0123456789")

        def num_numeric_values(text: str):
            num = 0
            words = text.split(" ")

            for word in words:
                if any(n in word for n in numeric_values):
                    num += 1

            return num

        data["num_single_quote_error"] = data["text"].map(
            lambda text: num_single_quote_error(text=text)
        )

        data["num_spacing_error"] = data["text"].map(
            lambda text: num_spacing_error(text=text)

        )

        data["num_space_absence_after_sentence_completion"] = data["text"].map(
            lambda text: num_space_absence_after_sentence_completion(text=text)
        )

        data["num_capitalized_words"] = data["text"].map(
            lambda text: num_capitalized_words(text=text)
        )

        data["num_capitalization_absence_after_sentence_completion"] = data["text"].map(
            lambda text: num_capitalization_absence_after_sentence_completion(text=text)
        )

        data["num_spelling_errors"] = data["text"].map(
            lambda text: num_spelling_errors(text=text)

        )

        data["num_words"] = data["text"].map(
            lambda text: num_words(text=text)
        )

        data["num_punctuations"] = data["text"].map(
            lambda text: num_punctuations(text=text)
        )

        data["num_numeric_values"] = data["text"].map(
            lambda text: num_numeric_values(text=text)
        )

        data["text"] = data["text"].map(
            lambda text: text.strip()
        )

        data["label"] = data["label"].map(
            lambda v: 0 if v == "fake" else 1)
        

        return data

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

        inp_train_tfrecord_dir = artifact_utils.get_split_uri(input_dict["examples"], "train")
        inp_val_tfrecord_dir = artifact_utils.get_split_uri(input_dict["examples"], "eval")
        inp_test_tfrecord_dir = artifact_utils.get_split_uri(input_dict["examples"], "test")

        output_artifact = output_dict["feature_engineered_examples"][0]
        output_artifact.split_names = artifact_utils.encode_split_names(["train", "eval", "test"])

        out_train_tfrecord_dir = artifact_utils.get_split_uri(output_dict["feature_engineered_examples"], "train")
        out_val_tfrecord_dir = artifact_utils.get_split_uri(output_dict["feature_engineered_examples"], "eval")
        out_test_tfrecord_dir = artifact_utils.get_split_uri(output_dict["feature_engineered_examples"], "test")

        input_data_dirs = [inp_train_tfrecord_dir, inp_val_tfrecord_dir, inp_test_tfrecord_dir]
        output_data_dirs = [out_train_tfrecord_dir, out_val_tfrecord_dir, out_test_tfrecord_dir]

        for (inp_data_dir, out_data_dir) in zip(input_data_dirs, output_data_dirs):
            inp_tfrecord_path = os.path.join(inp_data_dir, os.listdir(inp_data_dir)[0])

            dataset = self.convert_tfrecord_to_dataframe(tfrecord_file=inp_tfrecord_path)
            feature_dataset = self.generate_features(data=dataset)

            tf.io.gfile.makedirs(out_data_dir)

            self.dataframe_to_tfrecord(data=feature_dataset, tfrecord_dir=out_data_dir)



