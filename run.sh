#! /bin/bash
# Author: Sining Sun (Northwestern Polytechnical University, China)
# This recipe is used to do NN-PIT (LSTM, DNN or  BLSTM)




step=0

lists_dir=./lists/ #lists_dir is used to store some necessary files lists
mkdir -p $lists_dir
num_threads=12

tfrecords_dir=data/tfrecords/
gpu_id='0'
TF_CPP_MIN_LOG_LEVEL=1
rnn_num_layers=3
tr_batch_size=32

tt_batch_size=1
input_size=129
output_size=129

rnn_size=496
keep_prob=0.8
learning_rate=0.0005
halving_factor=0.7
decode=0
model_type=BLSTM

prefix=StandPsmPIT
assignment=def
name=${prefix}_${model_type}_${rnn_num_layers}_${rnn_size}_ReLU
save_dir=exp/$name/
data_dir=data/separated/${name}_${assignment}/
resume_training=false

# note: we want to use gender information, but we didn't use in this version. 
# but when we prepared our data, we stored the gender information (maybe useful in the future).
# wsj-train-spkrinfo.txt:  https://catalog.ldc.upenn.edu/docs/LDC93S6A/wsj0-train-spkrinfo.txt

# tfrecords are stored in data/tfrecords/{tr, cv, tt}_psm/

if [ $step -le 0 ]; then
    for x in tr cv tt; do
        python -u local/gen_tfreords.py --gender_list local/wsj0-train-spkrinfo.txt data/wav/wav8k/min/$x/ lists/${x}_wav.lst data/tfrecords/${x}_psm/ &

    done
    wait
fi
#####################################################################################################
#   NOTE for STEP 1:                                                                              ###
#       1. Make sure that you configure the RNN/data_dir/model_dir/ all rights                    ###
#####################################################################################################

if [ $step -le 1 ]; then
    
    echo "Start Traing RNN(LSTM or BLSTM) model."
    decode=0
    batch_size=25
    # Here, we made tfrecords list file for tr, cv and tt data. 
    # Make sure you have generated tfrecords files in $tfrecords_dir/{tr, cv, tt}_psm/
    # The list files name must be tr_tf.lst, cv_tf.lst and tt_tf.lst. We fixed them in run_lstm.py
    for x in tr tt cv; do
        find $tfrecords_dir/${x}_psm/ -iname "*.tfrecords" > $lists_dir/${x}_tf.lst
    done

    tr_cmd="python -u  run_lstm.py \
    --lists_dir=$lists_dir  --rnn_num_layers=$rnn_num_layers --batch_size=$batch_size --rnn_size=$rnn_size \
    --decode=$decode --learning_rate=$learning_rate --save_dir=$save_dir --data_dir=$data_dir --keep_prob=$keep_prob \
    --input_size=$input_size --output_size=$output_size  --assign=$assignment --resume_training=$resume_training \
    --model_type=$model_type --halving_factor=$halving_factor "

    echo $tr_cmd
    CUDA_VISIBLE_DEVICES=$gpu_id TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL $tr_cmd
fi
#####################################################################################################
#   NOTE for STEP 2:                                                                              ###
#       1. Make sure that you configure the RNN/data_dir/model_dir/ all rights                    ###
#####################################################################################################

if [ $step -le 2 ]; then
    
    echo "Start Decoding."
    decode=1
    batch_size=30
     tr_cmd="python -u  run_lstm.py --lists_dir=$lists_dir  --rnn_num_layers=$rnn_num_layers --batch_size=$batch_size --rnn_size=$rnn_size \
    --decode=$decode --learning_rate=$learning_rate --save_dir=$save_dir --data_dir=$data_dir --keep_prob=$keep_prob \
    --input_size=$input_size --output_size=$output_size  --assign=$assignment --resume_training=$resume_training \
    --model_type=$model_type --czt_dim=128"

    echo $tr_cmd
    CUDA_VISIBLE_DEVICES=$gpu_id TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL $tr_cmd
fi



