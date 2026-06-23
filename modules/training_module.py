import os
import json

import tensorflow as tf
import keras
import tensorflow_transform as tft

from tensorflow.keras.layers import Input

from tensorflow.keras.callbacks import TensorBoard

from modules.preprocessing_module import transform_feature_name
from models.get_news_authentication_model import GetNewsAuthenticationModel

#Loading Configs:

with open("configs/model_configs/training_configs.json","r") as file:
    BATCH_SIZE=json.load(file)["BATCH_SIZE"]


with open("configs/model_configs/feature_config.json","r") as file:
    features=json.load(file)

LABEL=features["LABEL"]
NUMERICAL_FEATURES=features["NUMERICAL_FEATURES"]
TEXT_FEATURE=features["TEXT_FEATURE"]


with open("configs/model_configs/text_model_config.json") as file:
    TEXT_SEQ_LEN=json.load(file)["text_seq_len"]


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

def get_serve_tf_examples_fn(model,
                             tf_transform_output,
                             tft_layer,
                             text_seq_len,
                             transformed_numerical_feature_names,
                             transformed_text_feature_name):

    @tf.function(input_signature=[tf.TensorSpec(shape=[None],dtype=tf.string,name="examples")])
    def serve_tf_examples_fn(serialized_tf_example):
        feature_spec=tf_transform_output.raw_feature_spec().copy()
        feature_spec.pop(LABEL,None)

        parsed_features=tf.io.parse_example(
            serialized_tf_example,
            feature_spec
        )

        transformed_features=tft_layer(parsed_features)

        model_inputs={}

        for feat in transformed_numerical_feature_names:
            tensor=transformed_features[feat]

            if isinstance(tensor,tf.SparseTensor):
                default_val=tf.constant(0,dtype=tf.float32)
                tensor=tf.sparse.to_dense(tensor,default_value=default_val)


            tensor=tf.cast(tensor,tf.float32)
            tensor=tf.reshape(tensor,[-1,1])
            tensor=tf.ensure_shape(tensor,[None,1])

            model_inputs[feat]=tensor

        text_feat=transformed_text_feature_name
        text_tensor=transformed_features[text_feat]

        if isinstance(text_tensor,tf.SparseTensor):
            default_val=tf.constant(0,dtype=tf.int64)
            text_tensor=tf.sparse.to_dense(text_tensor,default_value=default_val)

        text_tensor=tf.cast(text_tensor,tf.int64)
        text_tensor=tf.reshape(text_tensor,[-1,text_seq_len])
        text_tensor=tf.ensure_shape(text_tensor,[None,text_seq_len])

        model_inputs[text_feat]=text_tensor

        outputs=model(model_inputs)

        return {"outputs":outputs}

    return serve_tf_examples_fn



def run_fn(fn_args):
    tf.keras.backend.clear_session()

    tf_transform_output=tft.TFTransformOutput(fn_args.transform_output)

    #Loading Dataset:
    print(f"[INFO] Loading Dataset...")
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
    print(f"[INFO] Initializing Model...")
    model=GetNewsAuthenticationModel(vocab_size=vocab_size)

    #Training Model:
    print("Training Model...")
    history=model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=val_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[callback]
    )

    #Logging Training Details:
    print(f"[INFO] Logging Training Details...")
    training_details={}

    training_loss=history.history["loss"]
    training_details["avg_training_loss"]=round(sum(training_loss)/len(training_loss),2)

    validation_loss=history.history["val_loss"]
    training_details["avg_validation_loss"]=round(sum(validation_loss)/len(validation_loss),2)

    training_accuracy=history.history["accuracy"]
    training_details["avg_training_accuracy"]=round(sum(training_accuracy)/len(training_accuracy),2)

    validation_accuracy=history.history["val_accuracy"]
    training_details["avg_validation_accuracy"]=round(sum(validation_accuracy)/len(validation_accuracy),2)

    with open("configs/model_configs/training_details.json","w") as file:
        json.dump(training_details,file)


    print("Training Details: ")
    print(training_details)


    #Exporting Model:
    print(f"[INFO] Exporting Model...")
    transformed_numerical_feature_names=[transform_feature_name(feat) for feat in NUMERICAL_FEATURES]
    transformed_text_feature_name=transform_feature_name(TEXT_FEATURE)

    tft_layer=tf_transform_output.transform_features_layer()
    serve_fn=get_serve_tf_examples_fn(
        model=model,
        tf_transform_output=tf_transform_output,
        tft_layer=tft_layer,
        text_seq_len=TEXT_SEQ_LEN,
        transformed_numerical_feature_names=transformed_numerical_feature_names,
        transformed_text_feature_name=transformed_text_feature_name
    )

    # These are the exact lines that solved the most tedious bug. The bug took almost 5 days to be solved.
    export_archive = keras.export.ExportArchive()
    export_archive.track(model)
    export_archive.track(tft_layer)

    export_archive.add_endpoint(
        name="serving_default",
        fn=serve_fn,
    )

    export_archive.write_out(fn_args.serving_model_dir)
