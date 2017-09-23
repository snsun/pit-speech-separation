#! /bin/bash

# This recipe is used to train NN for speaker recognition


lists_dir=./tmp/lists/ #lists_dir is used to store some necessary files lists
mkdir -p $lists_dir
apply_cmvn=1
num_threads=12
tfrecords_dir=data/tfrecords/spknet/

gpu_id='0'
TF_CPP_MIN_LOG_LEVEL=1
rnn_num_layers=2
tr_batch_size=32
tt_batch_size=1
input_size=258
output_size=101
rnn_size=128
keep_prob=0.8
learning_rate=0.0005
halving_factor=0.7
decode=0
model_type=BLSTM
prefix=spknet
name=${prefix}_${model_type}_${rnn_num_layers}_${rnn_size}
save_dir=exp/spknet/$name/
data_dir=data/spknet/${name}/
resume_training=false
embedding_option=0

echo "Start Traing spknet"
batch_size=32
for x in tr cv; do
find $tfrecords_dir/${x}/ -iname "*.tfrecords" > $lists_dir/${x}.lst
done
tr_cmd="python run_spknet.py \
--lists_dir=$lists_dir  --rnn_num_layers=$rnn_num_layers --batch_size=$batch_size --rnn_size=$rnn_size \
--learning_rate=$learning_rate --save_dir=$save_dir --data_dir=$data_dir --keep_prob=$keep_prob \
--input_size=$input_size --output_size=$output_size  --resume_training=$resume_training \
--model_type=$model_type --halving_factor=$halving_factor --embedding_option=$embedding_option "

echo $tr_cmd
CUDA_VISIBLE_DEVICES=$gpu_id TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL $tr_cmd






