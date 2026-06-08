import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense,Concatenate

from typing_extensions import List,Dict,Text,Any


class NumericalFeatureProcessingLayer(Layer):
    def __init__(self,dense_layer_config:List[Dict[Text,Any]],**kwargs):
        super().__init__(**kwargs)

        self.dense_layer_config=dense_layer_config
        self.layers=[]

        for config in self.dense_layer_config:
            layer=Dense(
                units=config["units"],
                activation=config["activation"],
                kernel_initializer=config["kernel_initializer"],
                kernel_regularizer=config["kernel_regularizer"]
            )

            self.layers.append(layer)


    def call(self,inputs):
        x=Concatenate(axis=-1)(inputs)

        for layer in self.layers:
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