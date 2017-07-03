from grpc.beta import implementations
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS

n_samples = 10

host, port = FLAGS.server.split(':')
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

# Generate test data
x_data = np.arange(n_samples, step=1, dtype=np.float32)
x_data = np.reshape(x_data, (n_samples, 1))

# Send request
request = predict_pb2.PredictRequest()
request.model_spec.name = 'qiaozhi'
request.model_spec.signature_name = 'predict'
inputs_tensor = tf.contrib.util.make_tensor_proto(x_data, dtype=tf.float32)
request.inputs['inputs'].CopyFrom(inputs_tensor)
result = stub.Predict(request, 10.0)  # 10 secs timeout
print result
