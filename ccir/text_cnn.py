# -*- coding:UTF-8 -*-
import tensorflow as tf
import numpy as np
import random

class TextCNN(object):
    """
    A CNN for text similiarty.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def load_w2v(self, path, vocab_query, vocab_doc):
        with open(path,"r") as fin:
            for i in fin:
                line = i.strip().split(" ")
                word = line[0].decode("utf-8")
                vec = np.array([float(i) for i in line[1:]])
                index_query = list(vocab_query.transform([word]))[0][0]
                index_doc   = list(vocab_doc.transform([word]))[0][0]
                if index_query != 0 and index_query < self.query_vocab_size:
                    self.w2v_query[index_query] = vec
                if index_doc != 0 and index_doc < self.doc_vocab_size:
                    self.w2v_doc[index_doc] = vec
        #print self.w2v
        self.w2v_query = tf.Variable(self.w2v_query)
        self.w2v_query = tf.cast(self.w2v_query, tf.float32)
        self.w2v_doc = tf.Variable(self.w2v_doc)
        self.w2v_doc = tf.cast(self.w2v_doc, tf.float32)

    def __init__(
      self, query_length, doc_length, num_classes,
      embedding_size, hidden_neural_size, filter_sizes, num_filters, vocab_query, vocab_doc, l2_reg_lambda=0.0):
        # init w2v
        self.query_vocab_size = len(vocab_query.vocabulary_)
        self.doc_vocab_size = len(vocab_doc.vocabulary_)
        self.w2v_query = np.array([random.uniform(-1/self.query_vocab_size**0.5,1/embedding_size**0.5) for i in range(self.query_vocab_size*embedding_size)]).reshape(self.query_vocab_size,embedding_size)
        self.w2v_doc = np.array([random.uniform(-1/self.doc_vocab_size**0.5,1/embedding_size**0.5) for i in range(self.doc_vocab_size*embedding_size)]).reshape(self.doc_vocab_size,embedding_size)
        #print type(self.w2v)
        path = "./wiki.vec"
        self.load_w2v(path, vocab_query, vocab_doc)

        # Placeholders for input, output and dropout
        with tf.name_scope("input"):
            self.input_x_1 = tf.placeholder(tf.int32, [None, query_length], name="input_x_1")
            self.input_x_2 = tf.placeholder(tf.int32, [None, doc_length], name="input_x_2")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
            self.overlap = tf.placeholder(tf.float32, [None, 1], name="overlap")
            self.batch_size_query = tf.placeholder(tf.int32, [None], name="batch_size_query")
            self.batch_size_doc = tf.placeholder(tf.int32, [None], name="batch_size_doc")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        #print self.input_y
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        self.high = 80
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            w_query = tf.Variable(
                tf.random_uniform([self.query_vocab_size, embedding_size], -1.0, 1.0),
                name="w_query")
            w_doc = tf.Variable(
                tf.random_uniform([self.doc_vocab_size, embedding_size], -1.0, 1.0),
                name="w_doc")
            with tf.name_scope("look_up_table"):
                self.embedded_chars_1 = tf.nn.embedding_lookup(w_query, self.input_x_1, name="chars_1")
                self.embedded_chars_2 = tf.nn.embedding_lookup(w_doc, self.input_x_2, name="chars_2")
                self.embedded_fine_1 = tf.nn.embedding_lookup(self.w2v_query, self.input_x_1, name="fine_1")
                self.embedded_fine_2 = tf.nn.embedding_lookup(self.w2v_doc, self.input_x_2, name="fine_2")
            with tf.name_scope("expanded"):
                self.embedded_chars_expanded_1 = tf.expand_dims(self.embedded_chars_1, -1, name="chars_expand_1")
                self.embedded_chars_expanded_2 = tf.expand_dims(self.embedded_chars_2, -1, name="chars_expand_2")
                self.embedded_fine_expanded_1 = tf.expand_dims(self.embedded_fine_1, -1, name="fine_expand_1")
                self.embedded_fine_expanded_2 = tf.expand_dims(self.embedded_fine_2, -1, name="fine_expand_2")
            with tf.name_scope("cat_embedding"):
                self.embedded_expand_1 = tf.concat(3, [self.embedded_chars_expanded_1, self.embedded_fine_expanded_1], name="char_fine_1")
                self.embedded_expand_2 = tf.concat(3, [self.embedded_chars_expanded_2, self.embedded_fine_expanded_2], name="char_fine_2")
        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope("query"):
            with tf.name_scope("conv-maxpool"):
                # Convolution Layer
                filter_size = 3
                filter_shape = [filter_size, embedding_size, 2, num_filters]
                #                                          channel
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_expand_1, #[batch=?, in_height, in_width, in_channels]
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                conv = tf.nn.batch_normalization(conv, 0.001, 1.0, 0,1, 0.0001)
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                #h = tf.nn.elu(tf.nn.bias_add(conv, b), name="elu")
                # Maxpooling over the outputs
                self.pooled_query = tf.nn.max_pool(
                    h,
                    ksize=[1, query_length - filter_size + 1, 1, 1],
                    #ksize=[1, query_length, embedding_size, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
            
            with tf.name_scope("MP"):
                self.pooled_query = tf.nn.batch_normalization(self.pooled_query, 0.001, 1.0, 0,1, 0.0001)
                W = tf.Variable(tf.truncated_normal([num_filters, num_filters], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                self.query_h = tf.tanh(tf.nn.xw_plus_b(tf.squeeze(self.pooled_query), W, b, name="scores"))
            
            with tf.name_scope("tile"):
                tile_cnt = doc_length - filter_size + 1 - self.high + 1
                self.query = tf.tile(self.query_h, [1, tile_cnt], name="tile")
                self.query = tf.reshape(self.query, [-1, tile_cnt, num_filters], name="re")

        with tf.name_scope("doc"):
            with tf.name_scope("conv-maxpool"):
                # Convolution Layer
                filter_size = 3
                filter_shape = [filter_size, embedding_size, 2, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                #print self.embedded_expand_2
                #print W
                conv = tf.nn.conv2d(
                    self.embedded_expand_2,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                conv = tf.nn.batch_normalization(conv, 0.001, 1.0, 0,1, 0.0001)
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                #h = tf.nn.elu(tf.nn.bias_add(conv, b), name="elu")
                # Maxpooling over the outputs
                self.pooled_doc = tf.nn.max_pool(
                    h,
                    ksize=[1, self.high, 1, 1],
                    #ksize=[1, doc_length, embedding_size, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
            with tf.name_scope("conv-maxpool-1"):
                # Convolution Layer
                filter_shape = [1, 1, num_filters, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.pooled_doc, #[batch=?, in_height, in_width, in_channels]
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                conv = tf.nn.batch_normalization(conv, 0.001, 1.0, 0,1, 0.0001)
                self.doc = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                #print h
        
        with tf.name_scope("Hadamand"):
            #self.hadamand = tf.mul(self.query, tf.squeeze(self.doc))    
            #self.query = tf.expand_dims(self.query, -1, name="expand_query")
            self.doc = tf.squeeze(self.doc, [2])
            #self.doc = tf.expand_dims(self.doc, -1)
            self.hadamand = tf.mul(self.query, self.doc)
            self.hadamand = tf.reshape(self.hadamand, [-1, num_filters * (doc_length - filter_size + 1 - self.high + 1)]) 
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            """
            W = tf.get_variable("W", shape=[(doc_length - filter_size + 1 - self.high + 1) * num_filters, num_filters], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            self.scores = tf.nn.relu(tf.nn.xw_plus_b(self.hadamand, W, b, name="scores"))
            W_2 = tf.get_variable("W_2", shape=[num_filters, 3], initializer=tf.contrib.layers.xavier_initializer())
            b_2 = tf.Variable(tf.constant(0.1, shape=[3]), name="b_2")
            self.scores = tf.nn.xw_plus_b(self.scores, W_2, b_2, name="scores")
            """
            self.hadamand = tf.nn.batch_normalization(self.hadamand, 0.001, 1.0, 0,1, 0.0001)
            W = tf.get_variable("W", shape=[(doc_length - filter_size + 1 - self.high + 1) * num_filters, 3], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[3]), name="b")
            self.scores = (tf.nn.xw_plus_b(self.hadamand, W, b, name="scores"))

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            #losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            #losses = tf.square(self.scores - self.input_y)
            #losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            #self.loss = tf.reduce_mean(losses) #+ l2_reg_lambda * l2_loss
            #self.loss = self.contrastive_loss(self.input_y,self.distance)
            self.y_test = tf.nn.softmax(self.scores)  
            #step2:do cross_entropy  
            cross_entropy = -tf.reduce_sum(self.input_y * tf.log(tf.clip_by_value(self.y_test, 1e-10, 1.0)))  
            self.loss = tf.reduce_mean(cross_entropy) + l2_reg_lambda * l2_loss
            """
            epsilon = tf.constant(value=0.000001, shape=[num_classes])
            logits = self.scores + epsilon
            softmax = tf.nn.softmax(logits)
            cross_entropy = -tf.reduce_sum(self.input_y * tf.log(softmax), reduction_indices=[1])
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            tf.add_to_collection('losses', cross_entropy_mean)
            self.loss = tf.add_n(tf.get_collection('losses'), name='loss')
            """
        # Accuracy
        with tf.name_scope("accuracy"):
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            #correct_predictions = tf.equal(tf.round(self.scores), self.input_y)
            #self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
