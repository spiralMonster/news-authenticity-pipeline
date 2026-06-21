import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc


def CreateGRPCStub(host,port=8500):
    host_port=f"{host}:{port}"
    channel=grpc.insecure_channel(host_port)

    stub=prediction_service_pb2_grpc.PredictionServiceStub(channel)
    return stub