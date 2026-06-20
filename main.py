import os
from utils.get_final_dataset import GetFinalDataset
from utils.convert_data_to_tfrecords import ConvertDataToTFRecords

from models.get_news_authentication_model import GetNewsAuthenticationModel
from making_prediction_from_model_server.making_predictions_via_rest_api import MakePredictions

DATA_DIR=os.path.join(os.getcwd(),"data")

# TRUE_NEWS_DATA_PATH=os.path.join(DATA_DIR,"True.csv")
# FAKE_NEWS_DATA_PATH=os.path.join(DATA_DIR,"Fake.csv")
#
# GetFinalDataset(true_news_data_path=TRUE_NEWS_DATA_PATH,
#                 fake_news_data_path=FAKE_NEWS_DATA_PATH)

# FINAL_DATASET_PATH=os.path.join(DATA_DIR,"final_dataset.csv")
#
#
# TFRecord_path=os.path.join(DATA_DIR,"tfrecords","final_dataset.tfrecord")
#
# ConvertDataToTFRecords(
#     csv_data_path=FINAL_DATASET_PATH,
#     tfrecord_path=TFRecord_path
# )

# GetNewsAuthenticationModel()


text="""
Hey, this is my project and  I AM EPEBBFEF nfrn,n exciyed abmbfe   ,fhkehkjmejhksh swnm,I  cmenv
"""
response=MakePredictions(text=text)

print(response.json())
