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

text_model_config_copy=text_model_config.copy()

text_model_config["basic_text_processing_layer_config"]=basic_text_processing_layer_config
text_model_config["multi_lstm_dense_layer_config"]=multi_lstm_dense_layer_config


with open("configs/model_configs/numerical_feature_processing_layer_config.json","r") as file:
    numerical_feature_processing_layer_config=json.load(file)["dense_layer_config"]


with open("configs/model_configs/news_authentication_model_dense_layer_config.json","r") as file:
    news_authentication_model_dense_layer_config=json.load(file)["dense_layer_config"]


with open("configs/model_configs/optimizer_config.json","r") as file:
    optimizer_config=json.load(file)



def GetNewsAuthenticationModel(vocab_size:int=20000):
    inputs={}

    for feature_name in NUMERICAL_FEATURES:
        transformed_feature_name=transform_feature_name(feature_name)
        inputs[transformed_feature_name]=Input(
            shape=(1,),
            dtype=tf.float32,
            name=transformed_feature_name

        )


    transformed_numerical_feature_name=[k for k in inputs.keys()]


    transform_text_feature_name=transform_feature_name(TEXT_FEATURE)
    inputs[transform_text_feature_name]=Input(
            shape=(text_model_config["text_seq_len"],),
            dtype=tf.int64,
            name=transform_text_feature_name
        )



    #News Authentication Layer:
    text_model_config["vocab_size"]=vocab_size
    text_model_config_copy["vocab_size"]=vocab_size

    news_auth=NewsAuthenticationLayer(
        text_processing_model_config=text_model_config,
        numerical_feature_processing_model_config=numerical_feature_processing_layer_config,
        dense_layer_config=news_authentication_model_dense_layer_config,
        transformed_numerical_feature_name=transformed_numerical_feature_name,
        transformed_text_feature_name=transform_text_feature_name,
        name="news_authentication_layer"
    )(inputs)

    #Model Initialization:
    model=Model(
        inputs=inputs,
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

    # Log Model Details:
    model_details={}

    num_of_params=model.count_params()
    model_details["num_of_parameters"]=num_of_params

    model_size_in_MB=round((num_of_params*4)/(1024*1024),2)
    model_size=f"{model_size_in_MB} MB"
    model_details["model_size"]=model_size

    with open("configs/model_configs/model_details.json","w") as file:
        json.dump(model_details,file)


    print(f"[INFO] Model Details Logged.")


    # Save the changes made in text_model_config:
    with open("configs/model_configs/text_model_config.json","w") as file:
        json.dump(text_model_config_copy,file)

    return model



if __name__=="__main__":
    model=GetNewsAuthenticationModel()
