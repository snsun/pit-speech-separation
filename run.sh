#! /bin/bash

# This recipe is used to do NN-PIT (LSTM, DNN or  BLSTM)
# I am sorry that we use different tools to do different work in this recipe
#   1. Feature extraction: Matlab-> czt & fft features -> Kaldi ark(text)
#   2. Feature format transformation: Kaldi ark(txt) -> Kaldi ark(binary), scp -> Tensorflow records
#      We define the tf records format for our task, please see the codes for the details
#   3. Traing & Test model: Tensorflow

step=1

lists_dir=./lists/ #lists_dir is used to store some necessary files lists
mkdir -p $lists_dir
num_threads=12
tfrecords_dir=data/tfrecords/
gpu_id='0'
TF_CPP_MIN_LOG_LEVEL=1
rnn_num_layers=2
tr_batch_size=32
tt_batch_size=1
input_size=129
output_size=129
rnn_size=128
keep_prob=0.8
learning_rate=0.0005
halving_factor=0.7
decode=0
model_type=BLSTM
prefix=StandPsmPIT
assignment=def
name=${prefix}_${model_type}_${rnn_num_layers}_${rnn_size}
save_dir=exp/$name/
data_dir=data/separated/${name}_${assignment}/
resume_training=false

#ori_wav_path=/home/disk1/snsun/Workspace/tensorflow/kaldi/data/wsj0/create-speaker-mixtures/data/2speakers/wav8k/min/tt/mix/
ori_wav_path=/home/disk1/jqliu/LSTM_PIT/data/wav/mix_tt_8k/
  #rec_wav_path=data/wav/rec_deepcluster_${fs}_${assign}/
rec_wav_path=data/wav/rec/${name}_${assignment}/

#Step 0: extract features using matlab program. 
#    Note: You need to change the data_dir path and kaldi_feats_dir path in 
#          matlab_feats_extraction/extract_czt_fft_feats.m  accordng to your config;

if [ $step -le 0 ]; then
    for x in cv tt; do
        python -u local/gen_tfreords.py --gender_list local/wsj0-train-spkrinfo.txt data/wav/wav8k/min/$x/ lists/${x}_wav.lst data/tfrecords/${x}_psm/ &
    done
    wait
fi
#####################################################################################################
#   NOTE for STEP 3:                                                                              ###
#       1. Make sure that you configure the RNN/data_dir/model_dir/ all rights                    ###
#####################################################################################################

if [ $step -le 3 ]; then
    
    echo "Start Traing RNN(LSTM or BLSTM) model."
    decode=0
    batch_size=32
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
exit
#####################################################################################################
#   NOTE for STEP 4:                                                                              ###
#       1. Make sure that you configure the RNN/data_dir/model_dir/ all rights                    ###
#####################################################################################################

if [ $step -le 4 ]; then
    
    echo "Start Decoding."
    decode=1
    batch_size=30
     tr_cmd="python -u run_lstm.py --lists_dir=$lists_dir  --rnn_num_layers=$rnn_num_layers --batch_size=$batch_size --rnn_size=$rnn_size \
    --decode=$decode --learning_rate=$learning_rate --save_dir=$save_dir --data_dir=$data_dir --keep_prob=$keep_prob \
    --input_size=$input_size --output_size=$output_size  --assign=$assignment --resume_training=$resume_training \
    --model_type=$model_type "

    echo $tr_cmd
    CUDA_VISIBLE_DEVICES=$gpu_id TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL $tr_cmd
fi
#####################################################################################################
#   NOTE for STEP 5:                                                                              ###
#       1. Make sure you give the right ori_wav_path                                              ###
#       2. The reconstruted wav are stored in rec_wav_path                                        ###
#####################################################################################################


if [ $step -le 5 ]; then
    echo "Reconstructe the separated wav"
    mkdir -p $rec_wav_path
    . ./path.sh
    scp_list=$lists_dir/scp.lst
    find $data_dir -iname "*.scp" > $scp_list
    for line in `cat $scp_list`; do

        wavname=`basename -s .scp $line`
        w=`echo $wavname | awk -F '_' 'BEGIN{OFS="_"}{print $1,$2,$3,$4}'` 
        w=${w}.wav
        copy-feats scp:$line ark,scp:./tmp/tmp_enhan.ark,./tmp/tmp_enhan.scp || exit 1
        python  ./local/reconstruct_spectrogram.py ./tmp/tmp_enhan.scp ${ori_wav_path}/$w ${rec_wav_path}/${wavname} || exit 1
  done

rm ./tmp/tmp_enhan.* $scp_list
echo "Done OK!"
fi





