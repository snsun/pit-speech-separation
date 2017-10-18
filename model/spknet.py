#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
    Build the LSTM(BLSTM)  neural networks for speaker recognition.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import rnn
import numpy as np

class LSTM(object):
    """Build BLSTM or LSTM model for speaker recognition.
       If you use this module to train your module, make sure that 
       your prepare the right format data! 
 
    Attributes:
        config: Used to config our model
                config.input_size: feature (input) size;
                config.output_size: the final layer(output layer) size;
                config.rnn_size: the rnn cells' number
                config.batch_size: the batch_size for training
                config.rnn_num_layers: the rnn layers numbers
                config.keep_prob: the dropout rate
        inputs: [A,B], a T*(2D) matrix 
        labels: "two hot" target label
        lengths: the length  of every utterance
        infer: bool, if training(false) or test (true)
    """

    def __init__(self, config, inputs, labels, lengths, infer=False):
        self._inputs = inputs
        self._labels = labels
        self._lengths = lengths
        self._model_type = config.model_type
        if infer: # if infer, we prefer to run one utterance one time. 
            config.batch_size = 1
        outputs = self._inputs
        ## This first layer-- feed forward layer
        ## Transform the input to the right size before feed into RNN

        with tf.variable_scope('forward1'):
            outputs = tf.reshape(outputs, [-1, config.input_size])
            outputs = tf.layers.dense(outputs, units=config.rnn_size,
                                      activation=tf.nn.tanh,
                                      reuse=tf.get_variable_scope().reuse)
            outputs = tf.reshape(
                outputs, [config.batch_size,-1, config.rnn_size])
        
        ## Configure the LSTM or BLSTM model 
        ## For BLSTM, we use the BasicLSTMCell.For LSTM, we use LSTMCell. 
        ## You can change them and test the performance...

        if config.model_type.lower() == 'blstm': 
            with tf.variable_scope('blstm'):
                cell = tf.contrib.rnn.BasicLSTMCell(config.rnn_size)
                if not infer and config.keep_prob < 1.0:
                    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)

                lstm_fw_cell = tf.contrib.rnn.MultiRNNCell([cell] * config.rnn_num_layers)
                lstm_bw_cell = tf.contrib.rnn.MultiRNNCell([cell] * config.rnn_num_layers)
                lstm_fw_cell = _unpack_cell(lstm_fw_cell)
                lstm_bw_cell = _unpack_cell(lstm_bw_cell)
                result = rnn.stack_bidirectional_dynamic_rnn(
                    cells_fw = lstm_fw_cell,
                    cells_bw = lstm_bw_cell,
                    inputs=outputs,
                    dtype=tf.float32,
                    sequence_length=self._lengths)
                outputs, fw_final_states, bw_final_states = result
        if config.model_type.lower() == 'lstm':
            with tf.variable_scope('lstm'):
                def lstm_cell():
                    return tf.contrib.rnn.LSTMCell(
                        config.rnn_size, forget_bias=1.0, use_peepholes=True,
                        initializer=tf.contrib.layers.xavier_initializer(),
                        state_is_tuple=True, activation=tf.tanh)
                attn_cell = lstm_cell
                if not infer and config.keep_prob < 1.0:
                    def attn_cell():
                        return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=config.keep_prob)
                cell = tf.contrib.rnn.MultiRNNCell(
                    [attn_cell() for _ in range(config.rnn_num_layers)],
                    state_is_tuple=True)
                self._initial_state = cell.zero_state(config.batch_size, tf.float32)
                state = self.initial_state
                outputs, state = tf.nn.dynamic_rnn(
                    cell, outputs,
                    dtype=tf.float32,
                    sequence_length=self._lengths,
                    initial_state=self.initial_state)
                self._final_state = state
        
        ## Feed forward layer. Transform the RNN output to the right output size

        with tf.variable_scope('forward2'):
            if config.embedding_option == 0: #no embedding , frame by frame
                if self._model_type.lower() == 'blstm':
                    outputs = tf.reshape(outputs, [-1, 2*config.rnn_size])
                    in_size=2*config.rnn_size
                else:
                    outputs = tf.reshape(outputs, [-1, config.rnn_size])
                    in_size = config.rnn_size

            else:
                if self._model_type.lower() == 'blstm':
                    outputs = tf.reshape(outputs, [config.batch_size,-1, 2*config.rnn_size])
                    in_size=2*config.rnn_size
                else:
                    outputs = tf.reshape(outputs, [config.batch_size,-1, config.rnn_size])
                    in_size = config.rnn_size

                if config.embedding_option == 1:  #last frame embedding
                    #http://sqrtf.com/fetch-rnn-encoder-last-output-using-tf-gather_nd/
                    ind = tf.subtract(self._lengths, tf.constant(1))
                    batch_range = tf.range(config.batch_size)
                    indices = tf.stack([batch_range, ind], axis=1)

                    outputs = tf.gather_nd(outputs, indices)
                    self._labels = tf.reduce_mean(self._labels, 1) 
                elif config.embedding_option == 2: # mean pooing
                    outputs = tf.reduce_mean(outputs,1)         
                    self._labels = tf.reduce_mean(self._labels, 1) 
            out_size = config.output_size
            weights1 = tf.get_variable('weights1', [in_size, out_size],
            initializer=tf.random_normal_initializer(stddev=0.01))
            biases1 = tf.get_variable('biases1', [out_size],
            initializer=tf.constant_initializer(0.0))
            outputs = tf.matmul(outputs, weights1) + biases1
            if config.embedding_option == 0:
                outputs = tf.reshape(outputs, [config.batch_size, -1, out_size])
            self._outputs = tf.nn.sigmoid(outputs)
        # Ability to save the model
        self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=30)

        if infer: return
       
       
        # Compute loss(CE)
        self._loss=tf.losses.sigmoid_cross_entropy(self._labels, outputs)
        if tf.get_variable_scope().reuse: return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self.lr)
        #optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name='new_learning_rate')
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
       
    @property
    def inputs(self):
        return self._inputs_spk1,self._inputs_spk2

    @property
    def labels(self):
        return self._labels

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def loss(self):
        return self._loss

    @property
    def train_op(self):
        return self._train_op
    @property
    def outputs(self):
        return self._outputs


    @staticmethod
    def _weight_and_bias(in_size, out_size):
        # Create variable named "weights".
        weights = tf.get_variable('weights', [in_size, out_size],
            initializer=tf.random_normal_initializer(stddev=0.01))
        # Create variabel named "biases".
        biases = tf.get_variable('biases', [out_size],
            initializer=tf.constant_initializer(0.0))
        return weights, biases
def _unpack_cell(cell):
    if isinstance(cell,tf.contrib.rnn.MultiRNNCell):
        return cell._cells
    else:
        return [cell]
