import argparse
import os,sys
sys.path.append('.')

import multiprocessing
from io_funcs.signal_processing import audiowrite, stft, audioread
from local.utils import mkdir_p
import tensorflow as tf
import numpy as np
parser = argparse.ArgumentParser(description='Generate TFRecords files')
parser.add_argument('wavdir',
                    help='The parent dit of mix/s1/s2')
parser.add_argument('namelist',
                    help='The parent dit of mix/s1/s2')
                   
parser.add_argument('tfdir',
                    help='TFRecords files dir')
parser.add_argument('--gender_list','-g', default='', type=str,
                    help='The speekers gender list')

"""
    This file is used to generate tfrecords for gender-sensitive PIT 
    speech seperation. Every tfrecords file contains:
    inputs: [mix_speech_abs, max_speech_phase], shape:T*(fft_len*2)
    labels: [spker1_speech_abs, apker2_speech_abs], shape:T*(fft_len*2)
    gender: [spker1_gender, spker2_gender], shape:1*2 
"""

args = parser.parse_args()

wavdir = args.wavdir
tfdir = args.tfdir
namelist = args.namelist
mkdir_p(tfdir)
if args.gender_list is not '':
    apply_gender_info=True;
    gender_dict = {}
    fid = open(args.gender_list, 'r')
    lines = fid.readlines()
    fid.close()
    for line in lines:
        spk = line.strip('\n').split(' ')[0]
        gender = line.strip('\n').split(' ')[1]
        if gender.lower() == 'm':
            gender_dict[spk] = 1;
        else:
            gender_dict[spk] = 0

def make_sequence_example(inputs, labels, genders):
    """Returns a SequenceExample for the given inputs, labels and genders
    Args:
        inputs: A list of input vectors. Each input vector is a list of floats.
        labels: A list of label vectors. Each label vector is a list of floats.
        genders: A 1*2 vector [0, 1], [0,1], [1,1], [0, 0]
    Returns:
        A tf.train.SequenceExample containing inputs and labels(optional).
    """
    input_features = [
        tf.train.Feature(float_list=tf.train.FloatList(value=input_))
        for input_ in inputs]
    label_features = [
        tf.train.Feature(float_list=tf.train.FloatList(value=label))
        for label in labels]
    gender_features = [ tf.train.Feature(float_list=tf.train.FloatList(value=genders)) ]
    feature_list = {
        'inputs': tf.train.FeatureList(feature=input_features),
        'labels': tf.train.FeatureList(feature=label_features),
         'genders': tf.train.FeatureList(feature=gender_features)
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return tf.train.SequenceExample(feature_lists=feature_lists)



def gen_feats(wav_name):
    mix_wav_name = wavdir + '/mix/'+ wav_name
    s1_wav_name = wavdir + '/s1/' + wav_name
    s2_wav_name = wavdir + '/s2/' + wav_name
    
    mix_wav = audioread(mix_wav_name, offset=0.0, duration=None, sample_rate=8000)
    s1_wav = audioread(s1_wav_name, offset=0.0, duration=None, sample_rate=8000)
    s2_wav = audioread(s2_wav_name, offset=0.0, duration=None, sample_rate=8000)
    
    mix_stft = stft(mix_wav, time_dim=0, size=256, shift=128) 
    s1_stft = stft(s1_wav, time_dim=0, size=256, shift=128) 
    s2_stft = stft(s2_wav, time_dim=0, size=256, shift=128) 
    
    s1_gender = gender_dict[wav_name.split('_')[0][0:3]]
    s2_gender = gender_dict[wav_name.split('_')[2][0:3]]
    part_name = os.path.splitext(wav_name)[0]
    tfrecords_name = tfdir + '/' + part_name + '.tfrecords'
    print(tfrecords_name)
    with tf.python_io.TFRecordWriter(tfrecords_name) as writer:
        tf.logging.info(
            "Writing utterance %s" %tfrecords_name)
        mix_abs = np.abs(mix_stft)
        mix_angle = np.angle(mix_stft);
        s1_abs = np.abs(s1_stft); 
        s1_angle = np.angle(s1_stft)
        s2_abs = np.abs(s2_stft);
        s2_angle = np.angle(s2_stft)
        inputs = np.concatenate((mix_abs, mix_angle), axis=1)
        labels = np.concatenate((s1_abs*np.cos(mix_angle-s1_angle), s2_abs*np.cos(mix_angle-s2_angle)), axis = 1)
        gender = [s1_gender, s2_gender]
        ex = make_sequence_example(inputs, labels, gender)
        writer.write(ex.SerializeToString())
        
       

pool = multiprocessing.Pool(8)
workers= []
fid = open(namelist, 'r')
lines = fid.readlines()
fid.close()
for name in lines:
    name = name.strip('\n')
    workers.append(pool.apply_async(gen_feats, (name)))
    gen_feats(name)
pool.close()
pool.join()
