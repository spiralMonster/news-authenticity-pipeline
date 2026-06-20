import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Embedding

from models.model_for_text_feature.input_divider_layer import InputDividerLayer
from models.model_for_text_feature.multi_lstm_layer import MultiLSTMLayer

from typing_extensions import List,Dict,Text,Any


@tf.keras.utils.register_keras_serializable()
class TextProcessingLayer(Layer):
    """
    The Network to process the text features.
    """
    def __init__(
            self,
            num_models:int,
            vocab_size:int,
            embedding_dim:int,
            basic_text_processing_layer_config: Dict[Text,Any],
            multi_lstm_dense_layer_config: List[Dict[Text,Any]],
            **kwargs
    ):
        super().__init__(**kwargs)

        self.num_models=num_models
        self.vocab_size=vocab_size
        self.embedding_dim=embedding_dim
        self.basic_text_processing_layer_config=basic_text_processing_layer_config
        self.multi_lstm_dense_layer_config=multi_lstm_dense_layer_config

        self.supports_masking = True

        #Embedding Layer:
        self.embedding_layer=Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            trainable=True,
            name="embedding_layer"
        )

        #Input Dividing Layer:
        self.input_divider_layer=InputDividerLayer(
            num_models=self.num_models,
            name="input_divider_layer"
        )

        #Multi LSTM Layer:
        self.multi_lstm_layer=MultiLSTMLayer(
            num_models=self.num_models,
            basic_text_processing_layer_config=self.basic_text_processing_layer_config,
            dense_layer_config=self.multi_lstm_dense_layer_config,
            name="multi_lstm_layer"
        )



    def call(self,x):
        x=self.embedding_layer(x)
        x=self.input_divider_layer(x)
        x=self.multi_lstm_layer(x)

        return x


    def compute_output_shape(self,input_shape):
        last_dim=self.num_models*self.multi_lstm_dense_layer_config[-1]["units"]
        shape=(input_shape[0],last_dim)

        return shape


    def get_config(self):
        config=super().get_config()
        config.update({
            "num_models":self.num_models,
            "vocab_size":self.vocab_size,
            "embedding_dim":self.embedding_dim,
            "basic_text_processing_layer_config":self.basic_text_processing_layer_config,
            "multi_lstm_dense_layer_config":self.multi_lstm_dense_layer_config
        })

        return config
