#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import tensorflow.python.platform

import numpy as np
from six.moves import xrange  
import tensorflow as tf

import data_utils
from tensorflow.models.rnn.translate import seq2seq_model
#from tensorflow.python.platform import gfile

from flask import Flask,request

import jieba

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 80,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("in_vocab_size", 100000, "input vocabulary size.")
tf.app.flags.DEFINE_integer("out_vocab_size", 100000, "output vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "datas", "Data directory")       
tf.app.flags.DEFINE_string("train_dir", "datas", "Training directory.")

FLAGS = tf.app.flags.FLAGS

_buckets = [(5, 10), (10, 15), (20, 25)]

def create_model(session, forward_only):
    model = seq2seq_model.Seq2SeqModel(
        FLAGS.in_vocab_size, FLAGS.out_vocab_size, _buckets,                           
        FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,      
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,                       
        forward_only=forward_only)                                                    
  
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)        
    model.saver.restore(session, ckpt.model_checkpoint_path)                        
  
    return model
    
app = Flask(__name__)

sess = tf.Session()
model = create_model(sess, True)                         
model.batch_size = 1  
in_vocab_path = os.path.join(FLAGS.data_dir, "vocab_in.txt")     
out_vocab_path = os.path.join(FLAGS.data_dir, "vocab_out.txt" )
in_vocab, _ = data_utils.initialize_vocabulary(in_vocab_path) 
_, rev_out_vocab = data_utils.initialize_vocabulary(out_vocab_path)

@app.route('/', methods=['GET', 'POST'])

def server():
    sentence = request.args.get("abc")
    seg_list = jieba.cut(sentence)
    sentence_seg = " ".join(seg_list).encode("utf-8")
    token_ids = data_utils.sentence_to_token_ids(sentence_seg, in_vocab)   
    bucket_id = min([b for b in xrange(len(_buckets))
                       if _buckets[b][0] > len(token_ids)])               
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
        {bucket_id: [(token_ids, [])]}, bucket_id)
    _, _, output_logits = model.step(
        sess, encoder_inputs, decoder_inputs,      
        target_weights, bucket_id, True)
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]       
    if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]                      
    res = ("".join([rev_out_vocab[output] for output in outputs]))
    return res 

if __name__ == "__main__":
  app.run("0.0.0.0", port=5000)
