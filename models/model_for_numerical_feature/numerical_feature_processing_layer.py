import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense

from typing_extensions import List,Dict,Text,Any


@tf.keras.utils.register_keras_serializable()
class NumericalFeatureProcessingLayer(Layer):
    """
    Layer for processing Numerical Features.
    """
    def __init__(self,dense_layer_config:List[Dict[Text,Any]],**kwargs):
        super().__init__(**kwargs)

        self.supports_masking = True

        self.dense_layer_config=dense_layer_config

        self.dense_layers=[]

        for ind,config in enumerate(self.dense_layer_config):
            layer=Dense(
                units=config["units"],
                activation=config["activation"],
                kernel_initializer=config["kernel_initializer"],
                kernel_regularizer=config["kernel_regularizer"],
                name=f"numerical_feature_processing_layer_dense_{ind}"
            )

            self.dense_layers.append(layer)


    def call(self, inputs):
        x=inputs
        for layer in self.dense_layers:
            x=layer(x)

        return x


    def compute_output_shape(self,input_shape):
        shape=(input_shape[0],self.dense_layer_config[-1]["units"])

        return shape


    def get_config(self):
        config=super().get_config()
        config.update({
            "dense_layer_config":self.dense_layer_config
        })

        return config