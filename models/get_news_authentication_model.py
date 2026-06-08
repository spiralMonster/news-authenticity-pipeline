import json

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from modules.preprocessing_module import transform_feature_name

from models.news_authentication_layer import NewsAuthenticationLayer

# Load Feature Names:
with open("configs/model_configs/feature_config.json","r") as file:
    feature_config=json.load(file)


NUMERICAL_FEATURES=feature_config["NUMERICAL_FEATURES"]
TEXT_FEATURE=feature_config["TEXT_FEATURE"]
LABEL=feature_config["LABEL"]


#Loading Model Configs:
with open("configs/model_configs/basic_text_processing_layer_config.json","r") as file:
    basic_text_processing_layer_config=json.load(file)["basic_text_processing_layer_config"]


with open("configs/model_configs/multi_lstm_dense_layer_config.json","r") as file:
    multi_lstm_dense_layer_config=json.load(file)["dense_layer_config"]


with open("configs/model_configs/text_model_config.json","r") as file:
    text_model_config=json.load(file)

text_model_config["basic_text_processing_layer_config"]=basic_text_processing_layer_config
text_model_config["multi_lstm_dense_layer_config"]=multi_lstm_dense_layer_config


with open("configs/model_configs/numerical_feature_processing_layer_config.json","r") as file:
    numerical_feature_processing_layer_config=json.load(file)["dense_layer_config"]


with open("configs/model_configs/news_authentication_model_dense_layer_config.json","r") as file:
    news_authentication_model_dense_layer_config=json.load(file)["dense_layer_config"]


with open("configs/model_configs/news_authentication_model_optimizer_config.json","r") as file:
    optimizer_config=json.load(file)



def GetNewsAuthenticationModel():
    numerical_inputs=[]

    for feature in NUMERICAL_FEATURES:
        inp=Input(
            shape=(1,),
            dtype=tf.int64,
            name=transform_feature_name(feature)
        )

        numerical_inputs.append(inp)


    text_inp=Input(
        shape=(text_model_config["text_seq_len"],),
        dtype=tf.float32,
        name=transform_feature_name(TEXT_FEATURE)
    )

    #News Authentication Layer:
    news_auth=NewsAuthenticationLayer(
        text_processing_model_config=text_model_config,
        numerical_feature_processing_model_config=numerical_feature_processing_layer_config,
        dense_layer_config=news_authentication_model_dense_layer_config,
        name="news_authentication_layer"
    )(
        {
            "numerical_inputs":numerical_inputs,
            "text_inputs":text_inp
        }
    )

    #Model Initialization:
    model=Model(
        inputs=[numerical_inputs,text_inp],
        outputs=news_auth
    )

    #Model Compilation:
    optimizer=Adam(
        learning_rate=optimizer_config["learning_rate"],
        beta_1=optimizer_config["beta_1"],
        beta_2=optimizer_config["beta_2"],
        clipnorm=optimizer_config["clipnorm"]
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary(expand_nested=True)

    return model




if __name__=="__main__":
    model=GetNewsAuthenticationModel()









