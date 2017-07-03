Tensorflow Serving

1. need to install before:
https://tensorflow.github.io/serving/setup#prerequisites

2. url:
https://tensorflow.github.io/serving/serving_basic

3. how to run:
    run the serving:
        bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=test --model_base_path=test_serving/model
    build the client:
        bazel build //test_serving/model:client
    run the client:
        bazel-bin/test_serving/client
    (test_serving is in tensorflow_serving)

NOTICE:
1. ImportError: No module named 'backports'

   solution: pip install backports.weakref

2. client.py", line 26, in <module> inputs_tensor = tf.contrib.util.make_tensor_proto(x_data, dtype=tf.float32) .....tensorflow.python.framework.errors_impl.NotFoundError: /root/tensor_word_space/tensor_server/serving/bazel-bin/test_serving/client.runfiles/org_tensorflow/tensorflow/contrib/image/python/ops/_single_image_random_dot_stereograms.so: undefined symbol: _ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci

    solution: https://software.intel.com/en-us/articles/train-and-use-a-tensorflow-model-on-intel-architecture
    (1) vim bazel-bin/tensorflow_serving/example/mnist_saved_model.runfiles/org_tensorflow/tensorflow/contrib/image/__init__.py
    Comment-out (#) the following line as shown:
    (2) #from tensorflow.contrib.image.python.ops.single_image_random_dot_stereograms import single_image_random_dot_stereograms
    (3) Save and close __init__.py.
    (4) Try issuing the command again:
