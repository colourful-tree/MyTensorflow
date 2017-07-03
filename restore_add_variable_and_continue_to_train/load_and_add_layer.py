import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
from model import Test

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_actual = tf.placeholder(tf.float32, shape=[None, 10])

w1 = tf.Variable(tf.zeros([784,10]), "w1")
b1 = tf.Variable(tf.zeros([10]), "b1")

c1 = tf.matmul(x,w1) + b1
y_h = tf.nn.softmax(c1)
correct_prediction = tf.equal(tf.argmax(y_h,1), tf.argmax(y_actual,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#init = tf.initialize_all_variables()
with tf.Session() as sess:
    #sess.run(init)
    saver = tf.train.Saver()
    save_path = saver.restore(sess, "model/model.ckpt")
    m = Test()
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(m.cross_entropy_2)
    sess.run(tf.variables_initializer([m.w2, m.b2]))
    def train(batch_xs, batch_ys):
        pre = sess.run([c1], feed_dict={x: batch_xs, y_actual: batch_ys})
        #pre = tf.reshape(pre[0], [-1,10])
        _, acc = sess.run([train_step, m.accuracy], feed_dict={m.x : pre[0], m.y_actual : batch_ys})
        #acc = sess.run([m.accuracy], feed_dict={m.x : pre[0], m.y_actual : batch_ys})
        #print acc
    def evaluation(batch_xs, batch_ys):
        pre = sess.run([c1], feed_dict={x: batch_xs, y_actual: batch_ys})
        acc = sess.run([m.accuracy], feed_dict={m.x : pre[0], m.y_actual : batch_ys})
        return acc

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(10)
        train(batch_xs, batch_ys)
        #pre = sess.run([c1], feed_dict={x: batch_xs, y_actual: batch_ys})
        #pre = tf.convert_to_tensor(np.array(pre))
        #pre = tf.reshape(pre, [10,10])

        #print type(pre[0])
        #w2 = tf.Variable(tf.zeros([10,10]), "w2")
        #b2 = tf.Variable(tf.zeros([10]), "b2")
        #y_predict_2 = tf.nn.softmax(tf.matmul(pre, w2) + b2)
        #correct_prediction = tf.equal(tf.argmax(y_predict_2,1), tf.argmax(y_actual,1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        if(i%100==0):
            print evaluation(mnist.test.images, mnist.test.labels)
            #print pre
            #pre = sess.run([c1], feed_dict={x: mnist.test.images, y_actual: mnist.test.labels})
            #acc = sess.run([m.accuracy], feed_dict={m.x: pre[0], y_actual: mnist.test.labels})
            #print acc
