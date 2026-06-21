import tensorflow as tf
import os

EXPORT_DIR = "../serving_model_dir"
EXPORT_DIR=os.path.join(EXPORT_DIR,os.listdir(EXPORT_DIR)[0])

loaded = tf.saved_model.load(EXPORT_DIR)

infer = loaded.signatures["serving_default"]

example = tf.train.Example(
    features=tf.train.Features(
        feature={
            "text": tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[b"I wannna say somehting sbdjwvjmdnkjwhvk,ndm vklejfl"]
                )
            ),
            "num_single_quote_error": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[10])
            ),
            "num_spacing_error": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[0])
            ),
            "num_space_absence_after_sentence_completion": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[0])
            ),
            "num_capitalized_words": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[2])
            ),
            "num_capitalization_absence_after_sentence_completion": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[0])
            ),
            "num_spelling_errors": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[0])
            ),
            "num_punctuations": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[0])
            ),
            "num_numeric_values": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[2])
            ),
            "num_words": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[7])
            ),
        }
    )
)

serialized = example.SerializeToString()

result = infer(examples=tf.constant([serialized]))

print(result)



