#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang     Xiaomi

from __future__ import absolute_import

import sys
import pprint

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    sys.stdout.flush()
