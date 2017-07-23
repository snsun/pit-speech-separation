#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017  Xiaomi Corporation (author: Ke Wang)

"""Build the LSTM neural networks.

This module provides an example of definiting compute graph with tensorflow.
In order to make the code concise, we split different parts of graph
definition into functions. However, just doing so doesn't work, since every
time the functions are called, the graph would be extended by new code.
So we use properties to ensure the compute graph will be constructed only if the
function is called at first time, and then be stored for subsequent use.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

import tensorflow as tf
import numpy as np

class LSTM(object):
    """Build the feed forward fully connected neural networks.

    This class is a feed forward fully connected neural networks model. We
    split different definition into different functions for simplicity.

    Attributes:
        config: A tensorflow placeholder indicating the input of the model.
        inputs: A tensorflow placeholder indicating the output of the model.
        labels: The model's prediction result after feeding forward.
        lengths: The training step used to optimize the model.
    """

    def __init__(self, config, inputs_cmvn,inputs, labels1,labels2, lengths, infer=False):
        self._inputs = inputs_cmvn
        self._mixed = inputs
        self._labels1 = labels1
        self._labels2 = labels2
        self._lengths = lengths

        if infer:
            config.batch_size = 1

        outputs = self._inputs
        with tf.variable_scope('forward1'):
            outputs = tf.reshape(outputs, [-1, config.input_dim])
            outputs = tf.layers.dense(outputs, units=config.rnn_size,
                                      activation=tf.nn.tanh,
                                      reuse=tf.get_variable_scope().reuse)
            outputs = tf.reshape(
                outputs, [config.batch_size, -1, config.rnn_size])

        with tf.variable_scope('lstm'):
            def lstm_cell():
                return tf.contrib.rnn.LSTMCell(
                    config.rnn_size, forget_bias=1.0, use_peepholes=True,
                    initializer=tf.contrib.layers.xavier_initializer(),
                    state_is_tuple=True, activation=tf.tanh)
            attn_cell = lstm_cell
            if not infer and config.keep_prob < 1.0:
                def attn_cell():
                    return tf.contrib.rnn.DropoutWrapper(
                        lstm_cell(), output_keep_prob=config.keep_prob)
            cell = tf.contrib.rnn.MultiRNNCell(
                [attn_cell() for _ in range(config.rnn_num_layers)],
                state_is_tuple=True)
            self._initial_state = cell.zero_state(config.batch_size, tf.float32)
            state = self.initial_state
            outputs, state = tf.nn.dynamic_rnn(
                cell, outputs,
                dtype=tf.float32,
                sequence_length=self.lengths,
                initial_state=self.initial_state)
            self._final_state = state

        with tf.variable_scope('forward2'):
            outputs = tf.reshape(outputs, [-1, config.rnn_size])
	    in_size=config.rnn_size
	    out_size = config.output_dim
            weights1 = tf.get_variable('weights1', [in_size, out_size],
            initializer=tf.random_normal_initializer(stddev=0.01))
            biases1 = tf.get_variable('biases1', [out_size],
            initializer=tf.constant_initializer(0.0))
            weights2 = tf.get_variable('weights2', [in_size, out_size],
            initializer=tf.random_normal_initializer(stddev=0.01))
            biases2 = tf.get_variable('biases2', [out_size],
            initializer=tf.constant_initializer(0.0))
            mask1 = tf.nn.sigmoid(tf.matmul(outputs, weights1) + biases1)
            mask2 = tf.nn.sigmoid(tf.matmul(outputs, weights2) + biases2)
            self._activations1 = tf.reshape(
                mask1, [config.batch_size, -1, config.output_dim])
            self._activations2 = tf.reshape(
                mask2, [config.batch_size, -1, config.output_dim])

            self._cleaned1 = self._activations1*self._mixed
            self._cleaned2 = self._activations2*self._mixed
        # Ability to save the model
        self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=30)

        if infer: return

        # Compute loss(Mse)
        cost1 = tf.reduce_mean( tf.reduce_sum(tf.pow(self._cleaned1-self._labels1,2),1)
                               +tf.reduce_sum(tf.pow(self._cleaned2-self._labels2,2),1)
                               ,1) 
        cost2 = tf.reduce_mean( tf.reduce_sum(tf.pow(self._cleaned2-self._labels1,2),1)
                               +tf.reduce_sum(tf.pow(self._cleaned1-self._labels2,2),1)
                               ,1)    

        idx = tf.cast(cost1>cost2,tf.float32)
        self._loss = tf.reduce_sum(idx*cost2+(1-idx)*cost1)
       # self._loss = loss * config.output_dim * 0.5
        #self._loss = tf.maximum(loss1,loss2)
        if tf.get_variable_scope().reuse: return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name='new_learning_rate')
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
    def get_opt_output(self):
	cost1 = tf.reduce_sum(tf.pow(self._cleaned1-self._labels1,2),2)+tf.reduce_sum(tf.pow(self._cleaned2-self._labels2,2),2)
                           
        cost2 = tf.reduce_sum(tf.pow(self._cleaned2-self._labels1,2),2)+tf.reduce_sum(tf.pow(self._cleaned1-self._labels2,2),2)    

	idx = tf.slice(cost1, [0, 0], [1, -1]) > tf.slice(cost2, [0, 0], [1, -1])
	idx = tf.cast(idx, tf.float32)
	idx = tf.reduce_mean(idx,reduction_indices=0)
        idx = tf.reshape(idx, [tf.shape(idx)[0], 1])	
	x1 = self._cleaned1[0,:,:] * (1-idx) + self._cleaned2[0,:, :]*idx
	
	x2 = self._cleaned1[0,:,:]*idx + self._cleaned2[0,:,:]*(1-idx)
	row = tf.shape(x1)[0]
	col = tf.shape(x1)[1]
	x1 = tf.reshape(x1, [1, row, col])
	x2 = tf.reshape(x2, [1, row, col])
	return x1, x2
	 
    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def lengths(self):
        return self._lengths

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
    def activations(self):
        return self._activations

    @property
    def loss(self):
        return self._loss

    @property
    def train_op(self):
        return self._train_op

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        # Create variable named "weights".
        weights = tf.get_variable('weights', [in_size, out_size],
            initializer=tf.random_normal_initializer(stddev=0.01))
        # Create variabel named "biases".
        biases = tf.get_variable('biases', [out_size],
            initializer=tf.constant_initializer(0.0))
        return weights, biases
