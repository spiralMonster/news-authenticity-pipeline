import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import LSTM,Dense,Bidirectional

from typing_extensions import List,Dict



@tf.keras.utils.register_keras_serializable()
class BasicTextProcessingLayer(Layer):
    """
    Basic Network of LSTM and Dense layers to process the text.
    """
    def __init__(
            self,
            lstm_layer_config:List[Dict],
            dense_layer_config:List[Dict],
            **kwargs

    ):

        super().__init__(**kwargs)

        self.supports_masking = True

        self.lstm_layer_config=lstm_layer_config
        self.dense_layer_config=dense_layer_config

        self.lstm_layers=[]
        self.dense_layers=[]

        for ind,config in enumerate(self.lstm_layer_config):
            if config["bidirectional"]:
                layer=Bidirectional(
                    LSTM(
                        units=config["units"],
                        activation=config["activation"],
                        kernel_initializer=config["kernel_initializer"],
                        kernel_regularizer=config["kernel_regularizer"],
                        return_sequences=config["return_sequences"]

                    ),
                    name=f"basic_text_processing_layer_bidi_{ind}"
                )


            else:
                layer=LSTM(
                        units=config["units"],
                        activation=config["activation"],
                        kernel_initializer=config["kernel_initializer"],
                        kernel_regularizer=config["kernel_regularizer"],
                        return_sequences=config["return_sequences"],
                        name=f"basic_text_processing_layer_lstm_{ind}"

                )


            self.lstm_layers.append(layer)


        for ind,config in enumerate(self.dense_layer_config):
            layer=Dense(
                        units=config["units"],
                        activation=config["activation"],
                        kernel_initializer=config["kernel_initializer"],
                        kernel_regularizer=config["kernel_regularizer"],
                        name=f"basic_text_processing_layer_dense_{ind}"
                )

            self.dense_layers.append(layer)


    def call(self,x):
        for layer in self.lstm_layers:
            x=layer(x)

        for layer in self.dense_layers:
            x=layer(x)


        return x

    def compute_output_shape(self,input_shape):
        shape=(input_shape[0],self.dense_layer_config[-1]["units"])

        return shape


    def get_config(self):
        config=super().get_config()

        config.update(
            {
                "lstm_layer_config":self.lstm_layer_config,
                "dense_layer_config":self.dense_layer_config
            }
        )

        return config
