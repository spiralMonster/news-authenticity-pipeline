import requests


def GetModelMetaData(
        model_name="news_authenticator",
        host="localhost",
        port=8501
):
    url=f"http://{host}:{port}/v1/models/{model_name}/metadata"

    response=requests.get(url=url)
    response=response.json()

    return response


if __name__=="__main__":
    response=GetModelMetaData()
    print(response)