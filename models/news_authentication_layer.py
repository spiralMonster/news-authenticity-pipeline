import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense,Concatenate,Dropout

from models.model_for_text_feature.text_processing_network import TextProcessingLayer
from models.model_for_numerical_feature.numerical_feature_processing_layer import NumericalFeatureProcessingLayer

from typing_extensions import Dict,List,Text,Any


@tf.keras.utils.register_keras_serializable()
class NewsAuthenticationLayer(Layer):
    """
    The layer to authenticate the news article.
    """

    def __init__(
            self,
            text_processing_model_config:Dict[Text,Any],
            numerical_feature_processing_model_config: List[Dict[Text,Any]],
            dense_layer_config: List[Dict[Text,Any]],
            transformed_numerical_feature_name: List[str],
            transformed_text_feature_name: str,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.supports_masking = True

        self.text_processing_model_config=text_processing_model_config
        self.numerical_feature_processing_model_config=numerical_feature_processing_model_config
        self.dense_layer_config=dense_layer_config

        self.transformed_numerical_feature_name=transformed_numerical_feature_name
        self.transformed_text_feature_name=transformed_text_feature_name

        self.final_layers=[]

        # Text Processing Model:
        self.text_processing_model=TextProcessingLayer(
            num_models=self.text_processing_model_config["num_models"],
            vocab_size=self.text_processing_model_config["vocab_size"]+2,
            embedding_dim=self.text_processing_model_config["embedding_dim"],
            basic_text_processing_layer_config=self.text_processing_model_config["basic_text_processing_layer_config"],
            multi_lstm_dense_layer_config=self.text_processing_model_config["multi_lstm_dense_layer_config"],
            name="text_processing_layer"
        )

        # Numerical Features Processing Model:
        self.numerical_feature_processing_model=NumericalFeatureProcessingLayer(
            dense_layer_config=self.numerical_feature_processing_model_config,
            name="numerical_feature_processing_layer"
        )

        #Dense Layers:
        dropout_ind=0
        for ind,config in enumerate(self.dense_layer_config):
            layer = Dense(
                units=config["units"],
                activation=config["activation"],
                kernel_initializer=config["kernel_initializer"],
                kernel_regularizer=config["kernel_regularizer"],
                name=f"news_auth_layer_dense_{ind}"
            )

            if config["dropout"]:
                dropout_layer=Dropout(
                    rate=config["dropout_rate"],
                    name=f"news_auth_layer_dropout_{dropout_ind}"
                )

                self.final_layers.append(layer)
                self.final_layers.append(dropout_layer)

                dropout_ind += 1

            else:
                self.final_layers.append(layer)


        self.numerical_concatenate_layer=Concatenate(
            axis=-1,
            name="news_auth_layer_numerical_concatenate"
        )

        self.concatenate_layer=Concatenate(
            axis=-1,
            name="news_auth_layer_final_concatenate"
        )


    def call(
            self,
            inputs
    ):
        numerical_inputs=[]
        for feat in self.transformed_numerical_feature_name:
            inp=inputs[feat]
            numerical_inputs.append(inp)


        normalized_numerical_inputs=[]
        for inp in numerical_inputs:
            inp=tf.reshape(inp, [-1, 1])
            normalized_numerical_inputs.append(inp)

        numerical_inp_final=self.numerical_concatenate_layer(normalized_numerical_inputs)
        num_out=self.numerical_feature_processing_model(numerical_inp_final)

        text_inp=inputs[self.transformed_text_feature_name]
        text_out=self.text_processing_model(text_inp)

        out=self.concatenate_layer([num_out, text_out])

        for layer in self.final_layers:
            out=layer(out)

        return out


    def compute_output_shape(self,input_shape):
        first_shape = list(input_shape.values())[0]
        batch_size = first_shape[0]

        shape = (batch_size, self.dense_layer_config[-1]["units"])
        return shape


    def get_config(self):
        config=super().get_config()
        config.update({
            "text_processing_model_config":self.text_processing_model_config,
            "numerical_feature_processing_model_config":self.numerical_feature_processing_model_config,
            "dense_layer_config":self.dense_layer_config,
            "transformed_numerical_feature_name":self.transformed_numerical_feature_name,
            "transformed_text_feature_name":self.transformed_text_feature_name
        })

        return config
