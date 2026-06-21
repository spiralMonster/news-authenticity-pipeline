import os
import tensorflow as tf


def ConvertModelToTFlite():
    saved_model_parent_dir = "serving_model_dir"
    saved_model_dir = os.path.join(saved_model_parent_dir,os.listdir(saved_model_parent_dir)[0])


    converter=tf.lite.TFLiteConverter.from_saved_model(
        saved_model_dir
    )

    converter.optimizations=[
        tf.lite.Optimize.DEFAULT
    ]

    converter.target_spec.supported_ops=[
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    converter._experimental_lower_tensor_list_ops=False

    tflite_model=converter.convert()

    tflite_model_dir=os.path.join("tflite_model_dir","1")
    os.makedirs(tflite_model_dir,exist_ok=True)

    tflite_model_path=os.path.join(tflite_model_dir,"model.tflite")
    with open(tflite_model_path,"wb") as file:
        file.write(tflite_model)


    print(f"[INFO] Model converted into TFlite version.")
