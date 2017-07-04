#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import random
from text_nn import TextNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("data_file", "./data/ccir.test.json", "Data source")
tf.flags.DEFINE_integer("max_query_length", 20, "ccir.json max document length (default: 128)")
tf.flags.DEFINE_integer("max_doc_length", 500, "ccir.json max document length (default: 128)")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("hidden_neural_size", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob_train", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("dropout_keep_prob_dev", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 1000, "Batch Size (default: 64)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_text_1, x_text_2, y, overlap, p_id = data_helpers.load_data_and_labels(FLAGS.data_file)
# max_document_length = max([len(x.split(" ")) for x in x_text])
# Load vocabulary
vocab_processor_doc = learn.preprocessing.VocabularyProcessor.restore(os.path.join("/home/work/heqiaozhi/c/runs.all.1/", "vocab_doc"))
vocab_processor_query = learn.preprocessing.VocabularyProcessor.restore(os.path.join("/home/work/heqiaozhi/c/runs.all.1/", "vocab_query"))
x_1 = np.array(list(vocab_processor_query.transform(x_text_1)))
x_2 = np.array(list(vocab_processor_doc.transform(x_text_2)))
y = np.array(y)
overlap = np.array(overlap)
p_id = np.array(p_id)
# print p_id.tolist()
#x = x[11:20]
#y = y[11:20]
#print y
# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        nn = TextNN(
            query_length=FLAGS.max_query_length,
            doc_length=FLAGS.max_doc_length,
            num_classes=3,
            embedding_size=FLAGS.embedding_dim,
            hidden_neural_size=FLAGS.hidden_neural_size,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            vocab_query=vocab_processor_query,
            vocab_doc=vocab_processor_doc,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
        
        def evaluation_step(x1_batch, x2_batch, y_batch, overlap):
            bs_query = np.ones(len(x1_batch)) * FLAGS.max_query_length
            bs_doc = np.ones(len(x1_batch)) * FLAGS.max_doc_length
            feed_dict = {
                                 nn.input_x_1: x1_batch,
                                 nn.input_x_2: x2_batch,
                                 nn.input_y: y_batch,
                                 nn.batch_size_query: bs_query,
                                 nn.batch_size_doc: bs_doc,
                                 nn.overlap: overlap,
                                 nn.dropout_keep_prob: FLAGS.dropout_keep_prob_dev
            }
            loss, acc, pre, scores = sess.run([nn.loss, nn.accuracy, nn.predictions, nn.y_test], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print "summary:\t" + str(loss) + "\t" + str(acc)
            #print pre
            #print y_batch
            print pre.tolist()
            print scores.tolist()
            # print("acc {:g}".format(acc))
        
        saver = tf.train.Saver()
        #saver.restore(sess, "/home/work/heqiaozhi/c/runs.4.1/checkpoints/model-7700")
        saver.restore(sess, "/home/work/heqiaozhi/c/runs.all.1/bak/model-11300")
        #saver.restore(sess, "/home/work/heqiaozhi/c/runs/bak/model-10820")
        #evaluation_step(x_1, x_2, y, overlap)
        
        batches = data_helpers.eva_batch_iter(
            list(zip(x_1, x_2, y, overlap)), FLAGS.batch_size)
        for batch in batches:
            x_batch_1, x_batch_2, y_batch, over = zip(*batch)
            evaluation_step(x_batch_1, x_batch_2, y_batch, over)
            #evaluation_step(x_1, x_2, y, overlap)
            #current_step = tf.train.global_step(sess, global_step)
