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
tf.flags.DEFINE_string("user_file", "./Data/users.test.dat", "Data source")
tf.flags.DEFINE_string("movies_file", "./Data/movies.test.dat", "Data source")
tf.flags.DEFINE_string("ratings_file", "./Data/ratings.test.dat", "Data source")


# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 16, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 512, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 10, "Number of checkpoints to store (default: 5)")
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
x_user, x_movie, y = data_helpers.load_data_and_labels(FLAGS.user_file, FLAGS.movies_file, FLAGS.ratings_file)
# Load vocabulary
vocab_processor_user_id = learn.preprocessing.VocabularyProcessor.restore(os.path.join("/root/tensor_word_space/neural/movie/runs", "vocab_user_id"))
vocab_processor_user_gender = learn.preprocessing.VocabularyProcessor.restore(os.path.join("/root/tensor_word_space/neural/movie/runs", "vocab_user_gender"))
vocab_processor_user_age = learn.preprocessing.VocabularyProcessor.restore(os.path.join("/root/tensor_word_space/neural/movie/runs", "vocab_user_age"))
vocab_processor_user_occupation = learn.preprocessing.VocabularyProcessor.restore(os.path.join("/root/tensor_word_space/neural/movie/runs", "vocab_user_occupation"))
vocab_processor_movie_id = learn.preprocessing.VocabularyProcessor.restore(os.path.join("/root/tensor_word_space/neural/movie/runs", "vocab_movie_id"))
vocab_processor_movie_title = learn.preprocessing.VocabularyProcessor.restore(os.path.join("/root/tensor_word_space/neural/movie/runs", "vocab_movie_title"))
vocab_processor_movie_class = learn.preprocessing.VocabularyProcessor.restore(os.path.join("/root/tensor_word_space/neural/movie/runs", "vocab_movie_class"))

user_id = np.array(list(vocab_processor_user_id.transform([i[0] for i in x_user])))
user_gender = np.array(list(vocab_processor_user_gender.transform([i[1] for i in x_user])))
user_age = np.array(list(vocab_processor_user_age.transform([i[2] for i in x_user])))
user_occupation = np.array(list(vocab_processor_user_occupation.transform([i[3] for i in x_user])))
movie_id = np.array(list(vocab_processor_movie_id.transform([i[0] for i in x_movie])))
movie_title = np.array(list(vocab_processor_movie_title.transform([i[1] for i in x_movie])))
movie_class = np.array(list(vocab_processor_movie_class.transform([i[2] for i in x_movie])))
y = np.array(y)
movie_title_max_len = max([len(i[1].split(" ")) for i in x_movie ])
movie_class_max_len = max([len(i[2].split(" ")) for i in x_movie ])

np.random.seed()
shuffle_indices = np.random.permutation(np.arange(len(y)))
user_id = user_id[shuffle_indices][:10]
user_gender = user_gender[shuffle_indices][:10]
user_age = user_age[shuffle_indices][:10]
user_occupation = user_occupation[shuffle_indices][:10]
movie_id = movie_id[shuffle_indices][:10]
movie_title = movie_title[shuffle_indices][:10]
movie_class = movie_class[shuffle_indices][:10]
y = y[shuffle_indices][:10]
#print y
# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = MovieCNN(
            title_length=15,#movie_title_max_len,
            class_length=6,
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

        def evaluation_step(user_id, user_gender, user_age, user_occupation, movie_id, movie_title, movie_class, y):
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

            score = sess.run([cnn.distance],  feed_dict)
            print score[0].tolist()
            print y.tolist()
            #time_str = datetime.datetime.now().isoformat()
            #print("TRAIN {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
         
        saver = tf.train.Saver()

        saver.restore(sess, "runs/checkpoints/model-141000") 
        # sess.run(tf.global_variables_initializer())

        evaluation_step(user_id, user_gender, user_age, user_occupation, movie_id, movie_title, movie_class, y)
