import tensorflow as tf
import numpy as np


class MovieCNN(object):
    def __init__(
      self, title_length, class_length, num_classes, 
      user_id_size,
      user_gender_size,
      user_age_size,
      user_occupation_size,
      movie_id_size,
      movie_title_size,
      movie_class_size,
      embedding_size, 
      filter_sizes, 
      num_filters, 
      l2_reg_lambda=0.0):

        self.user_id = tf.placeholder(tf.int32, [None, 1], name="user_id")
        self.user_gender = tf.placeholder(tf.int32, [None, 1], name="user_gender")
        self.user_age = tf.placeholder(tf.int32, [None, 1], name="user_age")
        self.user_occupation = tf.placeholder(tf.int32, [None, 1], name="user_occupation")
        
        self.movie_id = tf.placeholder(tf.int32, [None, 1], name="movie_id")
        self.movie_title = tf.placeholder(tf.int32, [None, title_length], name="movie_title")
        self.movie_class = tf.placeholder(tf.int32, [None, class_length], name="movie_class")

        
        # Keeping track of l2 regularization loss (optional)
        #l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            w_user_id = tf.Variable(
                tf.random_uniform([user_id_size, embedding_size], -1.0, 1.0),
                name="w_user_id")
            self.embedded_user_id = tf.expand_dims(tf.nn.embedding_lookup(w_user_id, self.user_id), -1)
            w_user_gender = tf.Variable(
                tf.random_uniform([user_gender_size, embedding_size], -1.0, 1.0),
                name="w_user_gender")
            self.embedded_user_gender = tf.expand_dims(tf.nn.embedding_lookup(w_user_gender, self.user_gender), -1)
            w_user_age = tf.Variable(
                tf.random_uniform([user_age_size, embedding_size], -1.0, 1.0),
                name="w_user_age")
            self.embedded_user_age = tf.expand_dims(tf.nn.embedding_lookup(w_user_age, self.user_age), -1)
            w_user_occupation = tf.Variable(
                tf.random_uniform([user_occupation_size, embedding_size], -1.0, 1.0),
                name="w_user_occupation")
            self.embedded_user_occupation = tf.expand_dims(tf.nn.embedding_lookup(w_user_occupation, self.user_occupation), -1)

            w_movie_id = tf.Variable(
                tf.random_uniform([movie_id_size, embedding_size], -1.0, 1.0),
                name="w_movie_id")
            self.embedded_movie_id = tf.expand_dims(tf.nn.embedding_lookup(w_movie_id, self.movie_id), -1)
            w_movie_title = tf.Variable(
                tf.random_uniform([movie_title_size, embedding_size], -1.0, 1.0),
                name="w_movie_title")
            self.embedded_movie_title = tf.expand_dims(tf.nn.embedding_lookup(w_movie_title, self.movie_title), -1)
            w_movie_class = tf.Variable(
                tf.random_uniform([movie_class_size, embedding_size], -1.0, 1.0),
                name="w_movie_class")
            self.embedded_movie_class = tf.expand_dims(tf.nn.embedding_lookup(w_movie_class, self.movie_class), -1)
 
        #print self.embedded_user_id
        #print self.embedded_user_gender
        #print self.embedded_movie_id
        #print self.embedded_movie_title
        self.user =  tf.concat([self.embedded_user_id, 
                                self.embedded_user_gender,
                                self.embedded_user_age,
                                self.embedded_user_occupation], 2)
        #self.user = tf.reshape(self.user, [-1, embedding_size * 4])
        #print self.user
        #self.moive = tf.reshape(self.embedded_movie_id, [-1, embedding_size * 4])
        # Create a convolution + maxpool layer for movie_title
        with tf.name_scope("conv-maxpool-%s" % "movie"):
            # Convolution Layer
            movie_title_num_filters = embedding_size * 3 / 2
            filter_shape = [4, embedding_size, 1, movie_title_num_filters]
            #               4           128     channel  192
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[movie_title_num_filters]), name="b")
            conv = tf.nn.conv2d(
                    self.embedded_movie_title,   #[batch= ?, in_heiht = 20, in_width = 128, in_channels = 1]
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")# [?, title_length - 4 + 1, 1, 32]
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, title_length - 4 + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
            self.movie_title_pooled = tf.reshape(pooled, [-1, 1, movie_title_num_filters, 1])
        
        with tf.name_scope("conv-maxpool-%s" % "class"):
            # Convolution Layer
            movie_class_num_filters = embedding_size * 3 / 2
            filter_shape = [1, embedding_size, 1, movie_class_num_filters]
            #               1           128     channel  192
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[movie_class_num_filters]), name="b")
            conv = tf.nn.conv2d(
                    self.embedded_movie_class,   #[batch= ?, in_heiht = 20, in_width = 128, in_channels = 1]
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")# [?, class_length - 4 + 1, 1, 32]
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, class_length - 1 + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
            self.movie_class_pooled = tf.reshape(pooled, [-1, 1, movie_class_num_filters, 1])

        
        self.movie = tf.concat([self.embedded_movie_id, self.movie_title_pooled, self.movie_class_pooled], 2)
        #print self.movie
        
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs_1 = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                #                   3               128     channel  32
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.user,   #[batch= ?, in_heiht = 1, in_width = 512, in_channels = 1]
                    W,
                    strides=[1, 1, embedding_size, 1],
                    padding="VALID",
                    name="conv")#smae as slef.user[?, 1 , 512/128, 32]
                #print self.user
                #print conv
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                #print h
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, 1, 4, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_1.append(pooled)
        
        pooled_outputs_2 = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.movie,
                    W,
                    strides=[1, 1, embedding_size, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, 1, 4, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_2.append(pooled)
                
        # Combine all the pooled features
        #print pooled_outputs_1
        #print pooled_outputs_2
        #num_filters_total = num_filters * len(filter_sizes) * 2
        #self.h_pool = tf.concat(3, pooled_outputs_1 + pooled_outputs_2)
        #self.h_pool = tf.concat(3, pooled_outputs_2)
        #print self.h_pool
        #self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        #self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        self.h_pool_1 = tf.reshape(tf.concat(pooled_outputs_1, 3), [-1, num_filters])
        self.h_pool_2 = tf.reshape(tf.concat(pooled_outputs_2, 3), [-1, num_filters])
        self.h = tf.concat([self.h_pool_1, self.h_pool_2], 1)
        #print self.h
        #self.h_pool_2 = tf.concat(3, pooled_outputs_2)
        #self.h_pool_flat_2 = tf.reshape(self.h_pool_2, [-1, num_filters_total])
        # Add dropout
        #with tf.name_scope("dropout"):
        #    self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            #W = tf.get_variable(
            #    "W",
            #    shape=[num_filters_total, num_classes],
            #    initializer=tf.contrib.layers.xavier_initializer())
            #b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            #l2_loss += tf.nn.l2_loss(W)
            #l2_loss += tf.nn.l2_loss(b)
            #self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            #self.predictions = tf.argmax(self.scores, 1, name="predictions")
            #self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(self.h_pool_1,self.h_pool_2)),1,keep_dims=True))
            #print tf.reduce_sum(tf.mul(self.h_pool_1,self.h_pool_2))
            #self.distance = tf.reduce_sum(tf.mul(self.h_pool_1,self.h_pool_2),1,keep_dims=True)
            #self.distance = tf.div(self.distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.h_pool_flat_1),1,keep_dims=True)),tf.sqrt(tf.reduce_sum(tf.square(self.h_pool_flat_2),1,keep_dims=True))))
            #self.distance = tf.div(self.distance, tf.mul(tf.sqrt(tf.reduce_sum(tf.square(self.h_pool_1),1,keep_dims=True)),tf.sqrt(tf.reduce_sum(tf.square(self.h_pool_2),1,keep_dims=True)))) * 5 + 0.5
            #print self.h.shape[1]
            W = tf.Variable(tf.truncated_normal([int(self.h.shape[1]), 1], stddev = 0.1))
            b = tf.Variable(tf.zeros([1]))
            #print self.h
            #print W
            #self.distance = tf.nn.xw_plus_b(self.h, W , b, name="distance")
            self.distance = tf.matmul(self.h, W) + b
            #print self.distance
            #print "---"
            #print self.distance
            #self.distance = tf.reshape(self.distance, [-1], name="distance")
            #print self.distance
            # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            #self.losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.distance)
            #self.loss = tf.reduce_mean(self.losses)# + l2_reg_lambda * l2_loss
            self.loss = tf.reduce_mean(tf.square(self.distance - self.input_y))
            #self.loss = tf.square(self.distance - self.input_y)
            #print self.loss
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.round(self.distance), self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


cnn = MovieCNN(
            title_length=20,
            class_length=20,
            num_classes=1,
            user_id_size=20,
            user_gender_size=30,
            user_age_size=40,
            user_occupation_size=50,
            movie_id_size=60,
            movie_title_size=70,
            movie_class_size=8,
            embedding_size=128,
            filter_sizes=list(map(int, "1".split(","))),
            num_filters=32,
            )

