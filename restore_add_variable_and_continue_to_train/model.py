import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np

class Test(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 10])
        self.y_actual = tf.placeholder(tf.float32, shape=[None, 10])
        self.w2 = tf.Variable(tf.zeros([10,10]), "w2")
        self.b2 = tf.Variable(tf.zeros([10]), "b2")
        self.y_predict_2 = tf.nn.softmax(tf.matmul(self.x, self.w2) + self.b2)
        self.cross_entropy_2 = tf.reduce_mean(-tf.reduce_sum(self.y_actual * tf.log(self.y_predict_2), 1))
        self.correct_prediction = tf.equal(tf.argmax(self.y_predict_2,1), tf.argmax(self.y_actual,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
