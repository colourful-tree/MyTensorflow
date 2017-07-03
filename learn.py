import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def

#input_data = tf.placeholder(tf.int32, [batch_size, None])
with tf.variable_scope("scope"):
    input_x = tf.placeholder(tf.float32, [None, 1], name="input")
    #a = tf.get_variable("a", shape=[2, 4])
    w = tf.get_variable("w", shape=[1, 1], initializer=tf.contrib.layers.xavier_initializer())
    y = tf.matmul(input_x, w)
    

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#saver.save(sess, "tmp/model.ckpt")
print sess.run(y, {input_x: [[3.0]]})

builder = saved_model_builder.SavedModelBuilder("model/1")
signature = predict_signature_def(inputs={"inputs": input_x}, outputs={"outputs": y})
builder.add_meta_graph_and_variables(
      sess = sess, 
      tags=[tag_constants.SERVING], signature_def_map={"predict": signature}
      )#legacy_init_op=legacy_init_op)
builder.save()
