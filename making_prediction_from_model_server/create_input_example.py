import tensorflow as tf
import pandas as pd


def _bytes_feature(value):
    feat = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value])
    )

    return feat


def _int64_feature(value):
    feat=tf.train.Feature(
        int64_list=tf.train.Int64List(value=[value])
    )

    return feat



def CreateInputExample(df:pd.DataFrame):
    feature = {
        "num_single_quote_error": _int64_feature(value=df.iloc[0]["num_single_quote_error"]),
        "num_spacing_error": _int64_feature(value=df.iloc[0]["num_spacing_error"]),
        "num_space_absence_after_sentence_completion": _int64_feature(value=df.iloc[0]["num_space_absence_after_sentence_completion"]),
        "num_capitalized_words": _int64_feature(value=df.iloc[0]["num_capitalized_words"]),
        "num_capitalization_absence_after_sentence_completion": _int64_feature(value=df.iloc[0]["num_capitalization_absence_after_sentence_completion"]),
        "num_spelling_errors": _int64_feature(value=df.iloc[0]["num_spelling_errors"]),
        "num_punctuations": _int64_feature(value=df.iloc[0]["num_punctuations"]),
        "num_numeric_values": _int64_feature(value=df.iloc[0]["num_numeric_values"]),
        "num_words":_int64_feature(value=df.iloc[0]["num_words"]),
        "text":_bytes_feature(value=df.iloc[0]["text"].encode("utf-8"))
    }

    example=tf.train.Example(
        features=tf.train.Features(feature=feature)
    )

    example=example.SerializeToString()

    return example