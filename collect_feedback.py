import os
import csv

data_dir="data"
dataset_path=os.path.join(data_dir,"final_dataset.csv")


def CollectFeedback(text:str,label:str):
    with open(dataset_path,"a",newline="") as file:
        writer=csv.writer(file)
        writer.writerow([text,label])


    print(f"[INFO] Feedback collected.")


