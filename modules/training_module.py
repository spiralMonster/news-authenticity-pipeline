import os
import tensorflow as tf
import tensorflow_transform as tft

from tensorflow.keras.callbacks import TensorBoard

from modules.preprocessing_module import transform_feature_name
from models.get_news_authentication_model import GetNewsAuthenticationModel

BATCH_SIZE=32
LABEL="label"


def _gzip_reader_fn(filenames):
    reader=tf.data.TFRecordDataset(
        filenames,
        compression_type="GZIP"
    )

    return reader

def input_fn(file_pattern,tft_transform_output,batch_size=32):
    transformed_feature_spec=(
        tft_transform_output.transformed_feature_spec().copy()
    )

    dataset=tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=transform_feature_name(LABEL)

    )

    return dataset


def get_serve_tf_examples_fn(model,tf_transform_output):
    model.tft_layer=tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec=tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL)

        parsed_features=tf.io.parse_example(
            serialized_tf_examples,
            feature_spec
        )

        transformed_features=model.tft_layer(parsed_features)
        outputs=model(transformed_features)

        return {"outputs":outputs}

    return serve_tf_examples_fn



def run_fn(fn_args):
    tf_transform_output=tft.TFTransformOutput(fn_args.transform_output)

    #Loading Dataset:
    train_dataset=input_fn(fn_args.train_files,tf_transform_output)
    val_dataset=input_fn(fn_args.eval_files,tf_transform_output)

    #Tensorboard Setup:
    log_dir_path=os.path.join(os.path.dirname(fn_args.serving_model_dir),"logs")
    callback=TensorBoard(
        log_dir=log_dir_path,
        update_freq="batch"
    )

    #Find Vocab Size:
    vocab_size=tf_transform_output.vocabulary_size_by_name("vocab_file")

    #Loading Model:
    model=GetNewsAuthenticationModel(vocab_size=vocab_size)

    #Training Model:
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=val_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[callback]
    )

    signatures={
        'serving_default':
            get_serve_tf_examples_fn(
                model,
                tf_transform_output
            ).get_concrete_function(
                tf.TensorSpec(
                    shape=[None],
                    dtype=tf.string,
                    name='examples'
                )
            )
    }

    #Saving Model:
    tf.saved_model.save(
        model,
        fn_args.serving_model_dir,
        signatures=signatures
    )

