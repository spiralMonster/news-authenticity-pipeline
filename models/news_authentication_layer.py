import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense,Concatenate,Dropout

from models.model_for_text_feature.text_processing_network import TextProcessingLayer
from models.model_for_numerical_feature.numerical_feature_processing_layer import NumericalFeatureProcessingLayer

from typing_extensions import Dict,List,Text,Any

from modules.preprocessing_module import transform_feature_name



class NewsAuthenticationLayer(Layer):
    """
    The layer to authenticate the news article.
    """

    def __init__(
            self,
            text_processing_model_config:Dict[Text,Any],
            numerical_feature_processing_model_config: List[Dict[Text,Any]],
            dense_layer_config: List[Dict[Text,Any]],
            **kwargs
    ):
        super().__init__(**kwargs)

        self.text_processing_model_config=text_processing_model_config
        self.numerical_feature_processing_model_config=numerical_feature_processing_model_config
        self.dense_layer_config=dense_layer_config

        # Text Processing Model:
        self.text_processing_model=TextProcessingLayer(
            num_models=self.text_processing_model_config["num_models"],
            vocab_size=self.text_processing_model_config["vocab_size"]+2,
            embedding_dim=self.text_processing_model_config["embedding_dim"],
            basic_text_processing_layer_config=self.text_processing_model_config["basic_text_processing_layer_config"],
            multi_lstm_dense_layer_config=self.text_processing_model_config["multi_lstm_dense_layer_config"]
        )

        # Numerical Features Processing Model:
        self.numerical_feature_processing_model=NumericalFeatureProcessingLayer(
            dense_layer_config=self.numerical_feature_processing_model_config
        )

        #Dense Layers:
        self.dense_layers=[]
        for config in self.dense_layer_config:
            layer = Dense(
                units=config["units"],
                activation=config["activation"],
                kernel_initializer=config["kernel_initializer"],
                kernel_regularizer=config["kernel_regularizer"]
            )

            if config["dropout"]:
                dropout_layer=Dropout(config["dropout_rate"])

                self.dense_layers.append(layer)
                self.dense_layers.append(dropout_layer)

            else:
                self.dense_layers.append(layer)



    def call(self, inputs):
        text_feature_name=transform_feature_name("text")
        text_input=inputs[text_feature_name]

        numerical_inputs=[
            value
            for key,value in inputs.items()
            if key!=text_feature_name
        ]

        num_out=self.numerical_feature_processing_model(numerical_inputs)
        text_out=self.text_processing_model(text_input)
        out=Concatenate(axis=-1)([num_out, text_out])

        for layer in self.dense_layers:
            out=layer(out)

        return out



    def compute_output_shape(self,input_shape):
        shape=(input_shape[transform_feature_name("text")][0],self.dense_layer_config[-1]["units"])

        return shape


    def get_config(self):
        config=super().get_config()
        config.update({
            "text_processing_model_config":self.text_processing_model_config,
            "numerical_feature_processing_model_config":self.numerical_feature_processing_model_config,
            "dense_layer_config":self.dense_layer_config
        })

        return config
