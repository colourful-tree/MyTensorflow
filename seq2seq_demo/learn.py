import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense

n_steps = 5
n_input = 2
n_output = 1

def get_batch(batch_size):
    x = np.random.uniform(0, 1, size=[batch_size,n_steps, n_input])
    y = np.flip(x, axis=1)
    y = np.sum(y, axis=2)
    y = y.reshape((batch_size, n_steps, 1))
    
    seq = np.empty((batch_size), dtype=np.int)
    seq.fill(n_steps)
    return x, y, seq

x, y, seq = get_batch(3)
print x
print "---"
print y
print "---"
print seq
inp_seq_len = out_seq_len = n_steps
layers_stacked_count = 2  # Number of stacked recurrent cells, on the neural depth axis. 
n_hidden = 20

tf.reset_default_graph()

sess = tf.InteractiveSession()

# Placeholders
enc_inp = tf.placeholder(tf.float32, [None, inp_seq_len, n_input], name='encoder_input')
dec_target = tf.placeholder(tf.float32, [None, out_seq_len, n_output], name ='decoder_input')
#dec_targets = [tf.placeholder(tf.float32, [None, n_output]) for i in range(out_seq_len)]

target_length = tf.placeholder(tf.int32, [None], name='target_seq_length')
keep_prob = tf.placeholder(tf.float32, [], name='dropout_keep_prob')
sample_rate = tf.placeholder(tf.float32, [], name = 'sample_rate')

# ---- Encoder
enc_cells = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(n_hidden), output_keep_prob=keep_prob) for i in range(layers_stacked_count)]
enc_stk_cell = tf.contrib.rnn.MultiRNNCell(enc_cells)

encoded_outputs, encoded_states = tf.nn.dynamic_rnn(enc_stk_cell, enc_inp, dtype=tf.float32)

# ---- Decoder
dec_cells = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(n_hidden), output_keep_prob=keep_prob) for i in range(layers_stacked_count)]
dec_stk_cell = tf.contrib.rnn.MultiRNNCell(dec_cells)

#helper = tf.contrib.seq2seq.TrainingHelper(expect, expect_length) # Old
#n_in_layer = tf.layers.dense()
hlay = Dense(n_output, dtype=tf.float32)
#print hlay
print(type(hlay))
helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(dec_target, target_length, sample_rate, next_input_layer=hlay)

decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_stk_cell, helper=helper, initial_state=encoded_states)

decoder_outputs, final_decoder_state, length = tf.contrib.seq2seq.dynamic_decode(decoder)
decoder_logits = decoder_outputs.rnn_output

h = tf.contrib.layers.fully_connected(decoder_logits, n_output)

diff = tf.squared_difference(h, dec_target)
batch_loss = tf.reduce_sum(diff, axis=1)
loss = tf.reduce_mean(batch_loss)

optimiser = tf.train.AdamOptimizer(1e-3)
training_op = optimiser.minimize(loss)

init_op = tf.global_variables_initializer()

sess = tf.InteractiveSession()

init_op.run()

for e in range(10):
    for i in range(100):
        batch_x, batch_y ,seq = get_batch(100)
        training_op.run(feed_dict={enc_inp:batch_x, dec_target: batch_y, target_length:seq, keep_prob:0.5, sample_rate:0.5})
        
    print(loss.eval(feed_dict={enc_inp:batch_x, dec_target: batch_y, target_length:seq, keep_prob:1,sample_rate:0}))
