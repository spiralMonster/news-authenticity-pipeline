import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense,Concatenate

from models.model_for_text_feature.basic_text_processing_layer import BasicTextProcessingLayer

from typing_extensions import List,Dict


@tf.keras.utils.register_keras_serializable()
class MultiLSTMLayer(Layer):
    """
    The layer where the input is divided and then provided to different LSTM networks
    and then their outputs is aggregated.
    """
    def __init__(
            self,
            num_models:int,
            basic_text_processing_layer_config:Dict,
            dense_layer_config: List[Dict],
            **kwargs


    ):
        super().__init__(**kwargs)

        self.num_models=num_models
        self.basic_text_processing_layer_config=basic_text_processing_layer_config
        self.dense_layer_config=dense_layer_config

        self.basic_text_processing_layers=[]
        self.dense_layers=[]

        self.supports_masking=True
        self.concat_layer=Concatenate(
            axis=-1,
            name="multi_lstm_layer_concatenate"
        )



        for ind in range(self.num_models):
            layer=BasicTextProcessingLayer(
                lstm_layer_config=self.basic_text_processing_layer_config["lstm_layer_config"],
                dense_layer_config=self.basic_text_processing_layer_config["dense_layer_config"],
                name=f"multi_lstm_layer_model_{ind}"
            )

            self.basic_text_processing_layers.append(layer)


        for ind,config in enumerate(self.dense_layer_config):
            layer=Dense(
                units=config["units"],
                activation=config["activation"],
                kernel_initializer=config["kernel_initializer"],
                kernel_regularizer=config["kernel_regularizer"],
                name=f"multi_lstm_layer_dense_{ind}"
            )

            self.dense_layers.append(layer)



    def call(self,x):
        outputs=[]

        for (layer,inp) in zip(self.basic_text_processing_layers,x):
            out=layer(inp)
            outputs.append(out)

        out=self.concat_layer(outputs)

        for layer in self.dense_layers:
            out=layer(out)

        return out


    def compute_output_shape(self,input_shape):
        last_dim=self.dense_layer_config[-1]["units"]
        shape=(input_shape[0][0],last_dim)

        return shape

    def get_config(self):
        config=super().get_config()
        config.update({
            "num_models":self.num_models,
            "basic_text_processing_layer_config":self.basic_text_processing_layer_config,
            "dense_layer_config":self.dense_layer_config
        })

        return config


