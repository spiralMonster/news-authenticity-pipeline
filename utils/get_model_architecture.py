import tensorflow as tf

def GetModelArchitecture(model,model_name):
    tf.keras.utils.plot_model(
        model,
        to_file=f"snippets/model_snippets/{model_name}.png",
        show_shapes=True,
        expand_nested=True,
        dpi=200
    )