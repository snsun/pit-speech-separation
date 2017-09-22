#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017   Sining Sun 

"""Converts data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import struct
import sys

import numpy as np
import tensorflow as tf

sys.path.append('./')
from io_funcs.tfrecords_io import get_padded_batch_v2

tf.logging.set_verbosity(tf.logging.INFO)


def make_sequence_example(inputs,labels):
    """Returns a SequenceExample for the given inputs and labels(optional).
    """
    input_features = [
        tf.train.Feature(float_list=tf.train.FloatList(value=input_))
        for input_ in inputs]
    if labels is not None :
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


def main(_):
    fid = open(FLAGS.input_list, 'r')
    lines = fid.readlines()
    fid.close()

    fid = open(FLAGS.spk_list, 'r')
    spkers = fid.readlines()
    fid.close()
    num_spkers = len(spkers)
    spker_dict={}
    i=0
    for spker in spkers:
        spker_dict[spker.strip('\n')] = i
        i = i + 1
        
    file_list = [line.strip('\n') for line in lines]
    _, _, label1, label2, length = get_padded_batch_v2(
        file_list, 1, 257, 129, 1, 1, False)
    sess = tf.Session()
    sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for filename  in file_list:
        if coord.should_stop():
            break
        tmpname = filename.split('/')[-1]
        file_name = os.path.splitext(tmpname)[0]
        spker1 = file_name.split('_')[0][0:3]
        spker2 = file_name.split('_')[2][0:3]
        target = np.zeros([1, num_spkers])
        target[0, spker_dict[spker1]] = 1
        target[0, spker_dict[spker2]] = 1
        feats_spk1, feats_spk2 = sess.run([label1, label2])
        feats1 = np.concatenate((feats_spk1[0, :, :], feats_spk2[0, :, :]), axis=1)
        feats2 = np.concatenate((feats_spk2[0, :, :], feats_spk1[0, :, :]), axis=1)
        name1 = FLAGS.output_dir + '/'+file_name+'_1.tfrecords'
        name2 = FLAGS.output_dir + '/'+file_name+'_2.tfrecords'
        with tf.python_io.TFRecordWriter(name1) as writer:
            ex = make_sequence_example(feats1, target)
            writer.write(ex.SerializeToString())
        with tf.python_io.TFRecordWriter(name2) as writer:
            ex = make_sequence_example(feats2, target)
            writer.write(ex.SerializeToString())


if __name__ == '__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument(
        '--input_list',
        type=str,
        default='',
        help='The original tfrecords file list'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='',
        help='The output data dir'
    )
    parser.add_argument(
        '--spk_list',
        type=str, 
        default='',
        help='The spker id list'
    )

    FLAGS, unparsed = parser.parse_known_args()

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


