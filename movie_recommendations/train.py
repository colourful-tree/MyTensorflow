#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import random
from movie_cnn import MovieCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("user_file", "./Data/users.dat", "Data source")
tf.flags.DEFINE_string("movies_file", "./Data/movies.dat", "Data source")
tf.flags.DEFINE_string("ratings_file", "./Data/ratings.dat", "Data source")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 16, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 512, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 200, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 10, "Number of checkpoints to store (default: 5)")
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
x_user, x_movie, y = data_helpers.load_data_and_labels(FLAGS.user_file, FLAGS.movies_file, FLAGS.ratings_file)
"""
print len(x_user)
print len(x_movie)
print len(y)
print "---"
for i in range(len(x_user)):
    print str(x_user[i][0])+"\t"+str(x_user[i][1])+"\t"+str(x_user[i][2])+"\t"+str(x_user[i][3])+"\t"+str(x_movie[i][0])+"\t"+str(x_movie[i][1])+"\t"+str(x_movie[i][2])+"\t"+str(y[i][0])
"""
vocab_processor_user_id = learn.preprocessing.VocabularyProcessor(1)
user_id = np.array(list(vocab_processor_user_id.fit_transform([i[0] for i in x_user])))
vocab_processor_user_gender = learn.preprocessing.VocabularyProcessor(1)
user_gender = np.array(list(vocab_processor_user_gender.fit_transform([i[1] for i in x_user])))

vocab_processor_user_age = learn.preprocessing.VocabularyProcessor(1)
user_age = np.array(list(vocab_processor_user_age.fit_transform([i[2] for i in x_user])))

vocab_processor_user_occupation = learn.preprocessing.VocabularyProcessor(1)
user_occupation = np.array(list(vocab_processor_user_occupation.fit_transform([i[3] for i in x_user])))

vocab_processor_movie_id = learn.preprocessing.VocabularyProcessor(1)
movie_id = np.array(list(vocab_processor_movie_id.fit_transform([i[0] for i in x_movie])))

movie_title_max_len = max([len(i[1].split(" ")) for i in x_movie ])
vocab_processor_movie_title = learn.preprocessing.VocabularyProcessor(movie_title_max_len)
movie_title = np.array(list(vocab_processor_movie_title.fit_transform([i[1] for i in x_movie])))

movie_class_max_len = max([len(i[2].split(" ")) for i in x_movie ])
vocab_processor_movie_class = learn.preprocessing.VocabularyProcessor(movie_class_max_len)
movie_class = np.array(list(vocab_processor_movie_class.fit_transform([i[2] for i in x_movie])))

y = np.array(y)
print "---"
print movie_class_max_len

# Build vocabulary
# Write vocabulary
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

vocab_processor_user_id.save(os.path.join(out_dir, "vocab_user_id"))
vocab_processor_user_gender.save(os.path.join(out_dir, "vocab_user_gender"))
vocab_processor_user_age.save(os.path.join(out_dir, "vocab_user_age"))
vocab_processor_user_occupation.save(os.path.join(out_dir, "vocab_user_occupation"))

vocab_processor_movie_id.save(os.path.join(out_dir, "vocab_movie_id"))
vocab_processor_movie_title.save(os.path.join(out_dir, "vocab_movie_title"))
vocab_processor_movie_class.save(os.path.join(out_dir, "vocab_movie_class"))

"""
max_document_length = max([len(x.split(" ")) for x in (x_text_1+x_text_2)])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x_1 = np.array(list(vocab_processor.fit_transform(x_text_1)))
x_2 = np.array(list(vocab_processor.fit_transform(x_text_2)))


# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled_1 = x_1[shuffle_indices]
x_shuffled_2 = x_2[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train_1, x_dev_1 = x_shuffled_1[:dev_sample_index], x_shuffled_1[dev_sample_index:]
x_train_2, x_dev_2 = x_shuffled_2[:dev_sample_index], x_shuffled_2[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
"""

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = MovieCNN(
            title_length=movie_title_max_len,
            class_length=movie_class_max_len,
            num_classes=1,
            user_id_size=len(vocab_processor_user_id.vocabulary_),
            user_gender_size=len(vocab_processor_user_gender.vocabulary_),
            user_age_size=len(vocab_processor_user_age.vocabulary_),
            user_occupation_size=len(vocab_processor_user_occupation.vocabulary_),
            movie_id_size=len(vocab_processor_movie_id.vocabulary_),
            movie_title_size=len(vocab_processor_movie_title.vocabulary_),
            movie_class_size=len(vocab_processor_movie_class.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
                  
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(user_id, user_gender, user_age, user_occupation, movie_id, movie_title, movie_class, y):
            feed_dict = {
                cnn.user_id: user_id,
                cnn.user_gender: user_gender,
                cnn.user_age: user_age,
                cnn.user_occupation: user_occupation,
                cnn.movie_id: movie_id,
                cnn.movie_title: movie_title,
                cnn.movie_class: movie_class,
                cnn.input_y: y
            }
            _, step, loss, acc = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("TRAIN {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))
            #print("TRAIN {}: step {}, acc {:g}".format(time_str, step, acc))
            
        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(user_id, user_gender, user_age, user_occupation, movie_id, movie_title, movie_class, y)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            user_id_b, user_gender_b, user_age_b, user_occupation_b, movie_id_b, movie_title_b, movie_class_b, y_b = zip(*batch)
            train_step(user_id_b, user_gender_b, user_age_b, user_occupation_b, movie_id_b, movie_title_b, movie_class_b, y_b)
            current_step = tf.train.global_step(sess, global_step)
            #if current_step % FLAGS.evaluate_every == 0:
            #    print("\nEvaluation:")
            #    dev_step(x_dev_1, x_dev_2, y_dev)
            #    print("")
            
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
