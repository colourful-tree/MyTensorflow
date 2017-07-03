import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_actual = tf.placeholder(tf.float32, shape=[None, 10])

w1 = tf.Variable(tf.zeros([784,10]), "w1")
b1 = tf.Variable(tf.zeros([10]), "b1")

c1 = tf.matmul(x, w1) + b1
y_predict = tf.nn.softmax(c1)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual*tf.log(y_predict), 1))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_actual: batch_ys})
        if(i%100==0):
            print "accuracy:",sess.run(accuracy, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels})
    saver = tf.train.Saver()
    save_path = saver.save(sess, "model/model.ckpt")

print "save model!"
