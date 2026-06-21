import tensorflow as tf
from tensorflow_serving.apis import predict_pb2

from making_prediction_from_model_server.generate_features import GenerateFeatures
from making_prediction_from_model_server.create_input_example import CreateInputExample

from making_prediction_from_model_server.via_grpc.create_grpc_stub import CreateGRPCStub


def grpc_request(
        stub,
        data_sample,
        model_name="news_authenticator",
        signature_name="serving_default"
):
    request=predict_pb2.PredictRequest()

    request.model_spec.name=model_name
    request.model_spec.signature_name=signature_name

    request.inputs["examples"].CopyFrom(
        tf.make_tensor_proto(
            [data_sample],
            dtype=tf.string
        )
    )

    result=stub.Predict(request,10)

    response=result.outputs["outputs"]
    value=tf.make_ndarray(response)[0]

    return value



def MakePredictions(text:str):
    df=GenerateFeatures(text=text)
    print(f"[INFO] Features Generated.")

    inp_example=CreateInputExample(df=df)
    print(f"[INFO] Model Input Created.")

    stub=CreateGRPCStub(host="localhost",port=8500)
    print(f"[INFO] GRPC Stub Created.")

    print(f"[INFO] Making Model predictions.")
    result=grpc_request(
        stub=stub,
        data_sample=inp_example,
        model_name="news_authenticator",
        signature_name="serving_default"
    )

    return result




if __name__=="__main__":
    text="""
    Lewis hamiltoon has won the 8th wdc@!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!. He is the GOAT if mne n vnm mma a;; time.
    """

    result=MakePredictions(text=text)

    print(result)


