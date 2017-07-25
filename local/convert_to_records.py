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
import multiprocessing

import numpy as np
import tensorflow as tf

sys.path.append('./')
from io_funcs.tfrecords_io import make_sequence_example_two_labels

tf.logging.set_verbosity(tf.logging.INFO)

def convert_cmvn_to_numpy(inputs_cmvn, labels_cmvn):
  if FLAGS.labels_cmvn !='':
    """Convert global binary ark cmvn to numpy format."""
    tf.logging.info("Convert %s and %s to numpy format" % (
        inputs_cmvn, labels_cmvn))
    inputs_filename = FLAGS.inputs_cmvn
    labels_filename = FLAGS.labels_cmvn

    inputs = read_binary_file(inputs_filename, 0)
    labels = read_binary_file(labels_filename, 0)

    inputs_frame = inputs[0][-1]
    labels_frame = labels[0][-1]

    #assert inputs_frame == labels_frame

    cmvn_inputs = np.hsplit(inputs, [inputs.shape[1]-1])[0]
    cmvn_labels = np.hsplit(labels, [labels.shape[1]-1])[0]

    mean_inputs = cmvn_inputs[0] / inputs_frame
    stddev_inputs = np.sqrt(cmvn_inputs[1] / inputs_frame - mean_inputs ** 2)
    mean_labels = cmvn_labels[0] / labels_frame
    stddev_labels = np.sqrt(cmvn_labels[1] / labels_frame - mean_labels ** 2)

    cmvn_name = os.path.join(FLAGS.output_dir, "train_cmvn.npz")
    np.savez(cmvn_name,
             mean_inputs=mean_inputs,
             stddev_inputs=stddev_inputs,
             mean_labels=mean_labels,
             stddev_labels=stddev_labels)

    tf.logging.info("Write to %s" % cmvn_name)
  else :
    """Convert global binary ark cmvn to numpy format."""
    tf.logging.info("Convert %s to numpy format" % (
        inputs_cmvn))
    inputs_filename = FLAGS.inputs_cmvn

    inputs = read_binary_file(inputs_filename, 0)

    inputs_frame = inputs[0][-1]


    cmvn_inputs = np.hsplit(inputs, [inputs.shape[1]-1])[0]

    mean_inputs = cmvn_inputs[0] / inputs_frame
    stddev_inputs = np.sqrt(cmvn_inputs[1] / inputs_frame - mean_inputs ** 2)

    cmvn_name = os.path.join(FLAGS.output_dir, "train_cmvn.npz")
    np.savez(cmvn_name,
             mean_inputs=mean_inputs,
             stddev_inputs=stddev_inputs)

    tf.logging.info("Write to %s" % cmvn_name)


def read_binary_file(filename, offset=0):
    """Read data from matlab binary file (row, col and matrix).

    Returns:
        A numpy matrix containing data of the given binary file.
    """
    read_buffer = open(filename, 'rb')
    read_buffer.seek(int(offset), 0)
    header = struct.unpack('<xcccc', read_buffer.read(5))
    if header[0] != 'B':
        print("Input .ark file is not binary")
        sys.exit(-1)
    if header[1] == 'C':
        print("Input .ark file is compressed, exist now.")
        sys.exit(-1)

    rows = 0; cols= 0
    _, rows = struct.unpack('<bi', read_buffer.read(5))
    _, cols = struct.unpack('<bi', read_buffer.read(5))

    if header[1] == "F":
        tmp_mat = np.frombuffer(read_buffer.read(rows * cols * 4),
                                dtype=np.float32)
    elif header[1] == "D":
        tmp_mat = np.frombuffer(read_buffer.read(rows * cols * 8),
                                dtype=np.float64)
    mat = np.reshape(tmp_mat, (rows, cols))

    read_buffer.close()

    return mat

def process_in_each_thread(line, name, apply_cmvn, cmvn_for_labels):
    if name != 'test':
        utt_id, inputs_path, labels_path1, labels_path2 = line.strip().split()
        inputs_path, inputs_offset = inputs_path.split(':')
        labels_path1, labels_offset1 = labels_path1.split(':')
        labels_path2, labels_offset2 = labels_path2.split(':')
    else:
        utt_id, inputs_path = line.strip().split()
        inputs_path, inputs_offset = inputs_path.split(':')
    tfrecords_name = os.path.join(FLAGS.output_dir,
                                  utt_id + ".tfrecords")
    with tf.python_io.TFRecordWriter(tfrecords_name) as writer:
        tf.logging.info(
            "Writing utterance %s to %s" % (utt_id, tfrecords_name))
        inputs = read_binary_file(inputs_path, inputs_offset).astype(np.float64)
        if name != 'test':
            labels1 = read_binary_file(labels_path1, labels_offset1).astype(np.float64)
            labels2 = read_binary_file(labels_path2, labels_offset2).astype(np.float64)
        else:
            labels = None
        if apply_cmvn:
            cmvn = np.load(os.path.join(FLAGS.output_dir, "train_cmvn.npz"))
            inputs_cmvn = (inputs - cmvn["mean_inputs"]) / cmvn["stddev_inputs"]
            if labels1 is not None and cmvn_for_labels:
                labels1 = (labels1 - cmvn["mean_labels"]) / cmvn["stddev_labels"]
                labels2 = (labels2 - cmvn["mean_labels"]) / cmvn["stddev_labels"]
        ex = make_sequence_example_two_labels(inputs,inputs_cmvn, labels1,labels2)
        writer.write(ex.SerializeToString())

def convert_to(name, apply_cmvn=True, cmvn_for_labels=False):
    """Converts a dataset to tfrecords."""
    config_file = open(name)
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    pool = multiprocessing.Pool(FLAGS.num_threads)
    workers= []
    for line in config_file:
         workers.append(pool.apply_async(
         process_in_each_thread, (line, name, apply_cmvn, cmvn_for_labels)))
         #process_in_each_thread(line, name, apply_cmvn, cmvn_for_labels)
    pool.close()
    pool.join()

    config_file.close()


def main(unused_argv):
    # Convert to Examples and write the result to TFRecords.
    cmvn_for_labels = False
    if FLAGS.apply_cmvn:
      convert_cmvn_to_numpy(FLAGS.inputs_cmvn, FLAGS.labels_cmvn)
    if FLAGS.labels_cmvn != '':
      cmvn_for_labels = True

    #convert_to("myTest", apply_cmvn=True)
    convert_to(FLAGS.mapping_list, apply_cmvn=FLAGS.apply_cmvn,cmvn_for_labels=cmvn_for_labels)
    #convert_to("test", apply_cmvn=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mapping_list',
        type=str,
        default='',
        help='The kaldi mapping list prepared before')
    parser.add_argument(
        '--inputs_cmvn',
        type=str,
        default='data/inputs.cmvn',
        help='Inputs CMVN file name'
    )
    parser.add_argument(
        '--labels_cmvn',
        type=str,
        default='',
        help='Labels CMVN file name'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/tfrecords',
        help='Directory to write the converted result'
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        default='10',
        help='The number of threads to convert tfrecords files.'
    )
    parser.add_argument(
        '--apply_cmvn',
        type=int,
        default=1,
        help='Use cmvn of not'
    )
    FLAGS, unparsed = parser.parse_known_args()

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
