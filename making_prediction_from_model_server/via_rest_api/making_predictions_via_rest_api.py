import base64
import requests


from making_prediction_from_model_server.generate_features import GenerateFeatures
from making_prediction_from_model_server.create_input_example import CreateInputExample

def MakePredictions(text:str,port=8501,model_name="news_authenticator"):
    url=f"http://localhost:{port}/v1/models/{model_name}:predict"

    print(f"[INFO] Generating Features:")
    df=GenerateFeatures(text=text)

    print(f"[INFO] Creating Input Example for Model:")
    example=CreateInputExample(df=df)

    print(f"[INFO] Making Predictions:")
    payload={
        "instances":[
            {
                "b64":base64.b64encode(example).decode("utf-8")
            }
        ]
    }
    response=requests.post(url,json=payload)
    response=response.json()

    return response



if __name__=="__main__":
    text="Lewis hamilton is settttttttt to winnnnnn n  fbkjhfkjj thhe 8th WDCCCCCCCCCCCCijvkrnmcbjmbv cebjfjed,nj"

    result=MakePredictions(text=text)

    print(result.json())