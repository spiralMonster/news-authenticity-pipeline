import os
import pandas as pd

DATA_DIR=os.path.join(os.getcwd(),'data')


def GetFinalDataset(true_news_data_path:str,fake_news_data_path:str):

    true_news_data=pd.read_csv(true_news_data_path)
    true_news_data['label']="true"

    fake_news_data=pd.read_csv(fake_news_data_path)
    fake_news_data['label']="fake"

    final_data=pd.concat([true_news_data,fake_news_data],axis=0)
    final_data=final_data.sample(frac=1).reset_index(drop=True)

    final_data=final_data[['text','label']]

    final_data.to_csv(os.path.join(DATA_DIR,'final_dataset.csv'),index=False)

    print(f"[INFO] Final Dataset created successfully...")


