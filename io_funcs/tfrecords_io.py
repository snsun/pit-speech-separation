#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Sining Sun (NPU)


"""Utility functions for working with tf.train.SequenceExamples."""

import tensorflow as tf


def make_sequence_example(inputs, labels=None):
    """Returns a SequenceExample for the given inputs and labels(optional).
    Args:
        inputs: A list of input vectors. Each input vector is a list of floats.
        labels(optional): A list of label vectors. Each label vector is a list of floats.
    Returns:
        A tf.train.SequenceExample containing inputs and labels(optional).
    """
    input_features = [
        tf.train.Feature(float_list=tf.train.FloatList(value=input_))
        for input_ in inputs]
    if labels is not None:
        label_features = [
            tf.train.Feature(float_list=tf.train.FloatList(value=label))
            for label in labels]
        feature_list = {
            'inputs': tf.train.FeatureList(feature=input_features),
            'labels': tf.train.FeatureList(feature=label_features)
        }
    else:
        feature_list = {
            'inputs': tf.train.FeatureList(feature=input_features)
        }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return tf.train.SequenceExample(feature_lists=feature_lists)

def make_sequence_example_two_labels(inputs,inputs_cmvn, labels1, labels2):
    """Returns a SequenceExample for the given inputs and labels(optional).
    Args:
        inputs: A list of input vectors. Each input vector is a list of floats.
        labels1: A list of one kind labels
        labels2: A list of another lablels
    Returns:
        A tf.train.SequenceExample containing inputs and labels(optional).
    """
    input_cmvn_features = [
        tf.train.Feature(float_list=tf.train.FloatList(value=input_))
        for input_ in inputs_cmvn]
    input_features = [
        tf.train.Feature(float_list=tf.train.FloatList(value=input_))
        for input_ in inputs]
    if labels1 is not None and labels2 is not None:
        label_features1 = [
            tf.train.Feature(float_list=tf.train.FloatList(value=label))
            for label in labels1]
        label_features2 = [
            tf.train.Feature(float_list=tf.train.FloatList(value=label))
            for label in labels2]
        feature_list = {
            'inputs': tf.train.FeatureList(feature=input_features),
            'inputs_cmvn': tf.train.FeatureList(feature=input_cmvn_features),
            'labels1': tf.train.FeatureList(feature=label_features1),
            'labels2': tf.train.FeatureList(feature=label_features2),
        }
    else:
        feature_list = {
            'inputs': tf.train.FeatureList(feature=input_features)
        }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return tf.train.SequenceExample(feature_lists=feature_lists)


def get_padded_batch(file_list, batch_size, input_size, output_size,
                     num_enqueuing_threads=4, num_epochs=1, infer=False):
    """Reads batches of SequenceExamples from TFRecords and pads them.
    Can deal with variable length SequenceExamples by padding each batch to the
    length of the longest sequence with zeros.
    Args:
        file_list: A list of paths to TFRecord files containing SequenceExamples.
        batch_size: The number of SequenceExamples to include in each batch.
        input_size: The size of each input vector. The returned batch of inputs
            will have a shape [batch_size, num_steps, input_size].
        num_enqueuing_threads: The number of threads to use for enqueuing
            SequenceExamples.
    Returns:
        inputs: A tensor of shape [batch_size, num_steps, input_size] of floats32s.
        labels: A tensor of shape [batch_size, num_steps] of float32s.
        lengths: A tensor of shape [batch_size] of int32s. The lengths of each
            SequenceExample before padding.
    """
    file_queue = tf.train.string_input_producer(
        file_list, num_epochs=num_epochs, shuffle=(not infer))
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)

    if not infer:
        sequence_features = {
            'inputs': tf.FixedLenSequenceFeature(shape=[input_size],
                                             dtype=tf.float32),
            'labels': tf.FixedLenSequenceFeature(shape=[output_size],
                                             dtype=tf.float32)}

        _, sequence = tf.parse_single_sequence_example(
            serialized_example, sequence_features=sequence_features)

        length = tf.shape(sequence['inputs'])[0]

        capacity = 1000 + (num_enqueuing_threads + 1) * batch_size
        queue = tf.PaddingFIFOQueue(
            capacity=capacity,
            dtypes=[tf.float32, tf.float32, tf.int32],
            shapes=[(None, input_size), (None, output_size), ()])

        enqueue_ops = [queue.enqueue([sequence['inputs'],
                                      sequence['labels'],
                                      length])] * num_enqueuing_threads
    else:
        sequence_features = {
            'inputs': tf.FixedLenSequenceFeature(shape=[input_size],
                                             dtype=tf.float32)}

        _, sequence = tf.parse_single_sequence_example(
            serialized_example, sequence_features=sequence_features)

        length = tf.shape(sequence['inputs'])[0]

        capacity = 1000 + (num_enqueuing_threads + 1) * batch_size
        queue = tf.PaddingFIFOQueue(
            capacity=capacity,
            dtypes=[tf.float32, tf.int32],
            shapes=[(None, input_size), ()])

        enqueue_ops = [queue.enqueue([sequence['inputs'],
                                      length])] * num_enqueuing_threads

    tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))
    return queue.dequeue_many(batch_size)

def get_padded_batch_v2(file_list, batch_size, input_size, output_size,
                     num_enqueuing_threads=4, num_epochs=1, shuffle=True):
    """Reads batches of SequenceExamples from TFRecords and pads them.
    Can deal with variable length SequenceExamples by padding each batch to the
    length of the longest sequence with zeros.
    Args:
        file_list: A list of paths to TFRecord files containing SequenceExamples.
        batch_size: The number of SequenceExamples to include in each batch.
        input_size: The size of each input vector. The returned batch of inputs
            will have a shape [batch_size, num_steps, input_size].
        num_enqueuing_threads: The number of threads to use for enqueuing
            SequenceExamples.
    Returns:
        inputs: A tensor of shape [batch_size, num_steps, input_size] of floats32s.
        labels: A tensor of shape [batch_size, num_steps] of float32s.
        lengths: A tensor of shape [batch_size] of int32s. The lengths of each
            SequenceExample before padding.
    """
    file_queue = tf.train.string_input_producer(
        file_list, num_epochs=num_epochs, shuffle=shuffle)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)

    
    sequence_features = {
            'inputs': tf.FixedLenSequenceFeature(shape=[input_size],dtype=tf.float32),
            'inputs_cmvn': tf.FixedLenSequenceFeature(shape=[input_size],dtype=tf.float32),
            'labels1': tf.FixedLenSequenceFeature(shape=[output_size],dtype=tf.float32),
            'labels2': tf.FixedLenSequenceFeature(shape=[output_size],dtype=tf.float32),
    }

    _, sequence = tf.parse_single_sequence_example(
            serialized_example, sequence_features=sequence_features)

    length = tf.shape(sequence['inputs'])[0]

    capacity = 1000 + (num_enqueuing_threads + 1) * batch_size
    queue = tf.PaddingFIFOQueue(
            capacity=capacity,
            dtypes=[tf.float32, tf.float32,tf.float32, tf.float32, tf.int32],
            shapes=[(None, input_size),(None, input_size),(None, output_size), (None, output_size), ()])

    enqueue_ops = [queue.enqueue([sequence['inputs'],
                                      sequence['inputs_cmvn'],
                                      sequence['labels1'],
                                      sequence['labels2'],
                                      length])] * num_enqueuing_threads

    tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))
    return queue.dequeue_many(batch_size)   
