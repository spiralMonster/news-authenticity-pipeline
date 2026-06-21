from tensorflow_serving.apis import get_model_metadata_pb2

from making_prediction_from_model_server.via_grpc.create_grpc_stub import CreateGRPCStub

def GetModelMetaData(model_name="news_authenticator"):
    stub=CreateGRPCStub(host="localhost",port=8500)

    request=get_model_metadata_pb2.GetModelMetadataRequest()
    request.model_spec.name=model_name
    request.metadata_field.append("signature_def")

    response=stub.GetModelMetadata(request,5)

    result=response.metadata["signature_def"]
    result=result.SerializeToString().decode("utf-8","ignore")
    

    return result