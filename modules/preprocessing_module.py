import tensorflow as tf
from stopwords import get_stopwords
import tensorflow_transform as tft

STOPWORDS=get_stopwords("english")
MAX_SEQ_LEN=360

NUMERICAL_FEATURES=[
    "num_single_quote_error",
    "num_spacing_error",
    "num_space_absence_after_sentence_completion",
    "num_capitalized_words",
    "num_capitalization_absence_after_sentence_completion",
    "num_spelling_errors",
    "num_punctuations",
    "num_numeric_values",
    "num_words"
]

TEXT_FEATURES=[
    "text"
]

LABEL_FEATURE="label"


def transform_feature_name(feature:str):
    transformed_name=feature+"_transformed"

    return transformed_name

def convert_sparse_to_dense_tensors(x):
    default_value=''
    if type(x)==tf.SparseTensor:

        x = tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value)

    x=tf.squeeze(x, axis=1)
    return x


def min_max_scaling_of_numerical_feature(data):
    scaled_feature=tft.scale_to_0_1(data)

    return scaled_feature


def preprocess_text(text):
    #Removes URLS:
    text=tf.strings.regex_replace(
        text,
        r'((http|https)://)?(www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/[a-zA-Z0-9#./?=&%+-]*)?',
        ''
    )

    #Removes twitter URLS:
    text=tf.strings.regex_replace(
        text,
        r'pic.twitter.com/[a-zA-Z0-9]+',
        ''
    )

    #Removes punctuation marks:
    text=tf.strings.regex_replace(
        text,
        r'[[:punct:]]',
        ''
    )

    #Removes digits:
    text=tf.strings.regex_replace(
        text,
        r'\d+',
        ''
    )

    #Removes extra spaces:
    text=tf.strings.regex_replace(
        text,
        r'\s+',
        ' '
    )

    text=tf.strings.strip(text)
    text=tf.strings.lower(text)

    words=tf.strings.split(text)

    alpha_mask=tf.strings.regex_full_match(
        words,
        r'[a-z]+'
    )

    words=tf.ragged.boolean_mask(
        words,
        alpha_mask
    )

    #Removes Stop Words:
    STOPWORDS_TENSOR = tf.constant(STOPWORDS)
    stopword_mask = tf.reduce_all(
        tf.not_equal(
            tf.expand_dims(words, axis=-1),
            STOPWORDS_TENSOR
        ),
        axis=-1
    )

    words = tf.ragged.boolean_mask(
        words,
        stopword_mask
    )


    #Tokenize Text:
    tokenized_ids=tft.compute_and_apply_vocabulary(
        words,
        num_oov_buckets=1,
        top_k=20000,
        vocab_filename="vocab_file"
    )

    tokenized_ids=tokenized_ids+1

    #Pad Text:
    padded_tokens=tokenized_ids.to_tensor(
        default_value=0,
        shape=[None,MAX_SEQ_LEN]
    )

    return padded_tokens




def preprocessing_fn(inputs):
    outputs={}
    for feat in NUMERICAL_FEATURES:
        transformed_feature_name=transform_feature_name(feature=feat)
        scaled_feature=min_max_scaling_of_numerical_feature(data=inputs[feat])

        outputs[transformed_feature_name]=scaled_feature

    for feat in TEXT_FEATURES:
        transformed_feature_name=transform_feature_name(feat)
        inp=inputs[feat]
        inp=convert_sparse_to_dense_tensors(inp)
        processed_text=preprocess_text(inp)

        outputs[transformed_feature_name]=processed_text


    transformed_label_feature_name=transform_feature_name(LABEL_FEATURE)
    label_inp=inputs[LABEL_FEATURE]

    processed_label=tf.where(
        tf.equal(label_inp,"fake"),
        0,
        1
    )

    final_label=tf.cast(
        processed_label,
        tf.int64
    )

    outputs[transformed_label_feature_name]=final_label

    return outputs

