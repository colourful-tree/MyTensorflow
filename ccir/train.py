#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_nn import TextNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", 0.02, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data_file", "./data/ccir.demo.json", "Data source")
tf.flags.DEFINE_integer("max_query_length", 20, "ccir.json max document length (default: 128)")
tf.flags.DEFINE_integer("max_doc_length", 500, "ccir.json max document length (default: 128)")
tf.flags.DEFINE_integer("min_query_word_fre", 0, "min document word frequency (default: 5)")
tf.flags.DEFINE_integer("min_doc_word_fre",5, "min document word frequency (default: 5)")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("hidden_neural_size", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob_train", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("dropout_keep_prob_dev", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 5, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 20, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 20, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 15, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_text_1, x_text_2, y, overlap, p_id = data_helpers.load_data_and_labels(FLAGS.data_file)

# Build vocabulary
#max_document_length = max([len(x.split(" ")) for x in (x_text_1+x_text_2)])
vocab_processor_query = learn.preprocessing.VocabularyProcessor(FLAGS.max_query_length, FLAGS.min_query_word_fre)
vocab_processor_doc = learn.preprocessing.VocabularyProcessor(FLAGS.max_doc_length, FLAGS.min_doc_word_fre)
vocab_processor_query.fit(x_text_1)
vocab_processor_doc.fit(x_text_2)
x_1 = np.array(list(vocab_processor_query.transform(x_text_1)))
x_2 = np.array(list(vocab_processor_doc.transform(x_text_2)))
"""vocab_processor_query = preprocess()
vocab_processor_doc = preprocess()
vocab_processor_query.fit(x_text_1, FLAGS.min_query_word_fre)
vocab_processor_doc.fit(x_text_2, FLAGS.min_doc_word_fre)
x_1 = np.array(vocab_processor_query.transform(x_text_1))
x_2 = np.array(vocab_processor_doc.transform(x_text_2))"""
y = np.array(y)
overlap = np.array(overlap)
p_id = np.array(p_id)
#print x_1
#print x_2
#valid = vocab_processor.reverse(x_2)
# Randomly shuffle data

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled_1 = x_1[shuffle_indices]
x_shuffled_2 = x_2[shuffle_indices]
y_shuffled = y[shuffle_indices]
overlap_shuffled = overlap[shuffle_indices]
p_id_shuffled = p_id[shuffle_indices]
# Split train/test set
# TODO: This is very crude, should use cross-validation

dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train_1, x_dev_1 = x_shuffled_1[:dev_sample_index], x_shuffled_1[dev_sample_index:]
x_train_2, x_dev_2 = x_shuffled_2[:dev_sample_index], x_shuffled_2[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
overlap_train, overlap_dev = overlap_shuffled[:dev_sample_index], overlap_shuffled[dev_sample_index:]
p_id_train, p_id_dev = p_id_shuffled[:dev_sample_index], p_id_shuffled[dev_sample_index:]
"""
tmp = p_id_dev.tolist()
for i in tmp:
    print str(i[0]) + "," + str(i[1])
print end
"""
"""
x_train_1, x_dev_1 = x_shuffled_1, x_shuffled_1
x_train_2, x_dev_2 = x_shuffled_2, x_shuffled_2
y_train, y_dev = y_shuffled, y_shuffled
overlap_train, overlap_dev = overlap_shuffled, overlap_shuffled
"""
print("Vocabulary_query Size: {:d}".format(len(vocab_processor_query.vocabulary_)))
print("Vocabulary_doc Size: {:d}".format(len(vocab_processor_doc.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    print("started session")
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

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(0.5 * 1e-3)
        #optimizer = tf.train.RMSPropOptimizer(0.5 * 1e-3)
        #optimizer = tf.train.AdadeltaOptimizer(0.5 * 1e-3)
        grads_and_vars = optimizer.compute_gradients(nn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # early stop
        acc_record = []
        acc_best = 100000000
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", nn.loss)
        acc_summary = tf.summary.scalar("accuracy", nn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor_query.save(os.path.join(out_dir, "vocab_query"))
        vocab_processor_doc.save(os.path.join(out_dir, "vocab_doc"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x1_batch, x2_batch, y_batch, overlap, vocab_query, vocab_doc):
            """
            A single training step
            """
            """
            feed_dict = {
              nn.input_x: x_batch,
              nn.input_y: y_batch,
              nn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, nn.loss, nn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)
            """
            #print len(x1_batch)
            bs_query = np.ones(len(x1_batch)) * FLAGS.max_query_length
            bs_doc = np.ones(len(x1_batch)) * FLAGS.max_doc_length
            #x1_batch = vocab_query.pad(x1_batch, "query")
            #x2_batch = vocab_doc.pad(x2_batch, "doc")
            #print x1_batch
            feed_dict = {
                                 nn.input_x_1: x1_batch,
                                 nn.input_x_2: x2_batch,
                                 nn.input_y: y_batch,
                                 nn.batch_size_query: bs_query,
                                 nn.batch_size_doc: bs_doc,
                                 nn.overlap: overlap,
                                 nn.dropout_keep_prob: FLAGS.dropout_keep_prob_train
            }
            _, step, loss, accuracy, score, summaries = sess.run([train_op, global_step, nn.loss, nn.accuracy, nn.scores, train_summary_op],  feed_dict)
            time_str = datetime.datetime.now().isoformat()
            #print score
            print("TRAIN {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)


        def dev_step(x1_batch, x2_batch, y_batch, overlap, vocab_query, vocab_doc, current_step):
            """
            Evaluates model on a dev set
            """
            """
            feed_dict = {
              nn.input_x: x_batch,
              nn.input_y: y_batch,
              nn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, nn.loss, nn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
            """
            bs_query = np.ones(len(x1_batch)) * FLAGS.max_query_length
            bs_doc = np.ones(len(x1_batch)) * FLAGS.max_doc_length
            #x1_batch = vocab_query.pad(x1_batch, "query")
            #x2_batch = vocab_doc.pad(x2_batch, "doc")
            feed_dict = {
                                 nn.input_x_1: x1_batch,
                                 nn.input_x_2: x2_batch,
                                 nn.input_y: y_batch,
                                 nn.batch_size_query: bs_query,
                                 nn.batch_size_doc: bs_doc,
                                 nn.overlap: overlap,
                                 nn.dropout_keep_prob: FLAGS.dropout_keep_prob_dev
            }
            step, loss, accuracy, summaries = sess.run([global_step, nn.loss, nn.accuracy, dev_summary_op],  feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("DEV {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            dev_summary_writer.add_summary(summaries, step)
            global acc_record
            global acc_best
            k = 8
            print acc_record
            while len(acc_record) >= k:
                acc_record.pop(0)
            acc_record.append(loss)
            if loss < acc_best:
                acc_best = loss
                #saver.save(sess, "runs/best-model.ckpt")
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                return False
            print acc_record
            if loss > 4000:
                return False
            if loss - acc_best > 300:
                return True
            else:
                return False

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train_1, x_train_2, y_train, overlap_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch_1, x_batch_2, y_batch, over = zip(*batch)
            train_step(x_batch_1, x_batch_2, y_batch, over, vocab_processor_query, vocab_processor_doc)
            current_step = tf.train.global_step(sess, global_step)
            is_overfit = False
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                is_overfit = dev_step(x_dev_1, x_dev_2, y_dev, overlap_dev, vocab_processor_query, vocab_processor_doc, current_step)

            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
            if is_overfit:
                break
