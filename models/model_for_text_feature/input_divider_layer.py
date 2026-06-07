import tensorflow as tf
from tensorflow.keras.layers import Layer


class InputDividerLayer(Layer):
    """
    The layer to divide input so that it is fed to different LSTM networks.
    """
    def __init__(self,num_models,**kwargs):
        super().__init__(**kwargs)

        self.num_models=num_models


    def call(self,x):
        words_per_text=x.shape[1]
        factor=words_per_text//self.num_models
        embedding_dim=x.shape[2]

        inputs=tf.transpose(
            tf.reshape(
                x,
                (-1,self.num_models,factor,embedding_dim)
            ),
            perm=[1,0,2,3]
        )

        inputs=tf.cast(inputs,tf.float32)

        return inputs

    def compute_output_shape(self,input_shape):
        words_per_text = input_shape[1]
        factor = words_per_text // self.num_models
        shape=(self.num_models,input_shape[0],factor,input_shape[2])

        return shape

    def get_config(self):
        config=super().get_config()
        config.update(
            {
                "num_models":self.num_models
            }
        )

        return config



