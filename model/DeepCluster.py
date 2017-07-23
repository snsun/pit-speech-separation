#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Build the deep clustering neural networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import rnn
import numpy as np
from sklearn.cluster import KMeans

class LSTM(object):
    def __init__(self, config, inputs_cmvn,mixed, labels1,labels2,lengths, infer=False):
        self._inputs = inputs_cmvn
        self._mixed = mixed
        self._labels1 = labels1
        self._labels2 = labels2
        self._lengths = lengths
        self.f = tf.shape(self._inputs)[1]  
        if infer:
            config.batch_size = 1
         
        outputs = self._inputs
        with tf.variable_scope('forward1'):
            outputs = tf.reshape(outputs, [-1, config.input_dim])
            outputs = tf.layers.dense(outputs, units=config.rnn_size,
                                      activation=tf.nn.tanh,
                                      reuse=tf.get_variable_scope().reuse)
            outputs = tf.reshape(
                outputs, [config.batch_size,-1, config.rnn_size])
        self.temp1=outputs
 	with tf.variable_scope('blstm'):
            cell = tf.contrib.rnn.BasicLSTMCell(config.rnn_size)
            lstm_fw_cell = tf.contrib.rnn.MultiRNNCell([cell] * config.rnn_num_layers)
            lstm_bw_cell = tf.contrib.rnn.MultiRNNCell([cell] * config.rnn_num_layers)
            lstm_fw_cell = _unpack_cell(lstm_fw_cell)
            lstm_bw_cell = _unpack_cell(lstm_bw_cell)
        #    if not infer and config.keep_prob < 1.0:
         #           lstm_fw_cell=tf.contrib.rnn.DropoutWrapper(
          #              lstm_fw_cell, output_keep_prob=config.keep_prob)
           #         lstm_bw_cell=tf.contrib.rnn.DropoutWrapper(
            #            lstm_bw_cell, output_keep_prob=config.keep_prob)
            result = rnn.stack_bidirectional_dynamic_rnn(
			cells_fw = lstm_fw_cell,
			cells_bw = lstm_bw_cell,
			inputs=outputs,
			dtype=tf.float32,
			sequence_length=self._lengths)
	    outputs, fw_final_states, bw_final_states = result
            self.temp2=outputs
        with tf.variable_scope('forward2'):
            outputs = tf.reshape(outputs, [-1, 2*config.rnn_size])
            self.temp3=outputs
	    in_size=2*config.rnn_size
            out_size = config.embedding_size*config.output_dim
            weights1 = tf.get_variable('weights1', [in_size, out_size],
            initializer=tf.random_normal_initializer(stddev=0.01))
            biases1 = tf.get_variable('biases1', [out_size],
            initializer=tf.constant_initializer(0.0))
            self._V = tf.nn.sigmoid(tf.matmul(outputs, weights1) + biases1)
	    self._V = tf.reshape(self._V,[config.batch_size,-1,config.output_dim*config.embedding_size])

        

        idx = tf.cast(self._labels1>self._labels2,'int32')
        idx = tf.reshape(idx,[-1])
        idx = tf.reshape(idx,[tf.shape(idx)[0],1])
        Y = tf.concat([1-idx,idx],1)
        self._V = tf.cast(self._V,'float32')
	Y = tf.cast(Y,'float32')
        Y = tf.reshape(Y,[config.batch_size,-1,config.input_dim*2])
         
        sum=0
	step=100
        overlap = 50
        for i in range(0,2000,50):
	    tempV = self._V[:, i:i+step, :]
	    tempY = Y[:, i:i+step, :]
	    for j in range(32):
	        tV = tempV[j,:,:]
		tV = tf.reshape(tV,[-1,config.embedding_size])
		tY = tempY[j,:,:]
		tY = tf.reshape(tY,[-1,2])
		sum+=tf.reduce_sum(tf.pow(tf.matmul(tV,tf.transpose(tV)),2))+ tf.reduce_sum(tf.pow(tf.matmul(tY,tf.transpose(tY)),2))-2*tf.reduce_sum(tf.pow(tf.matmul(tf.transpose(tV),tY),2))
       
        self._loss = sum
        # Ability to save the model
        self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=30)
        if infer: return
        # Compute loss(Mse)
        if tf.get_variable_scope().reuse: return
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self.lr)
        #optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name='new_learning_rate')
        self._lr_update = tf.assign(self._lr, self._new_lr)
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
    def get_cleaned(self):
	kmeans = KMeans(2)
        mask = kmeans.fit_predict(self._V)
        mask = tf.convert_to_tensor(mask,'int32')
        mask = tf.reshape(mask,[config.batch_size, -1, config.output_dim])
        cleaned1 = mask*self._mixed
        cleaned2 = (1-mask)*self._mixed
        return cleaned1,cleaned2
	 
    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels1,self._labels2

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
    
def _unpack_cell(cell):
    if isinstance(cell,tf.contrib.rnn.MultiRNNCell):
	return cell._cells
    else:
	return [cell]

