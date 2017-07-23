#!/bin/bash
#
# Copyright 2017    Ke Wang     Xiaomi
#
# Train LSTM model using TensorFlow for speech enhancement

set -euo pipefail

stage=2
data=data
fs=8k
config=zoom_fft_list
exp=exp_${fs}_zoomfft/lstm
batch_size=32
decode=1
keep_prob=0.5
input_dim=257
output_dim=129
embedding_size=40
lr=0.000015625
gpu_id=1
assign='def'
data_dir=data/wsj0/wsj0_blstm_zoomfft_${fs}_${assign}/
if [ $decode -eq 1 ]; then
 batch_size=1
else
 batch_size=32
fi

# Prepare data
if [ $stage -le 0 ]; then
  python misc/get_train_val_scp.py
fi

# Prepare TFRecords format data
if [ $stage -le 1 ]; then
  [ ! -e "$data/tfrecords" ] && mkdir -p "$data/tfrecords"
  [ ! -e "$data/tfrecords/train" ] && mkdir -p "$data/tfrecords/train"
  [ ! -e "$data/tfrecords/val" ] && mkdir -p "$data/tfrecords/val"
  [ ! -e "$data/tfrecords/test" ] && mkdir -p "$data/tfrecords/test"
  python utils/convert_to_records_parallel.py \
    --data_dir=$data/raw/cmvn \
    --output_dir=$data/tfrecords \
    --config_dir=$config \
    --num_thread=12
fi

# Train LSTM model
if [ $stage -le 2 ]; then
  echo "Start train LSTM model."
  CUDA_VISIBLE_DEVICES=$gpu_id TF_CPP_MIN_LOG_LEVEL=1 python     run_lstm_8k.py \
    --config_dir=$config  --rnn_num_layers=2 --batch_size=$batch_size --decode=$decode \
    --learning_rate=$lr --save_dir=$exp --data_dir=$data_dir --keep_prob=$keep_prob \
    --input_dim=$input_dim --output_dim=$output_dim --embedding_size=$embedding_size --assign=$assign
fi
if [ $stage -le 3 ]; then

  . ./path.sh 
  scp_list=tmp.scp  #tmp file. will be rmoved 
  ori_wav_path=/home/disk1/jqliu/LSTM_PIT/data/wav/mix_tt_8k  
  #rec_wav_path=data/wav/rec_deepcluster_${fs}_${assign}/
  rec_wav_path=data/wav/rec_zoom_fft_${fs}_${assign}
  echo "Reconstructing time domain separated speech signal \n"
  echo "Store the reconstructed wav to $rec_wav_path \n"
  mkdir -p $rec_wav_path
  find $data_dir -iname "*.scp" > $scp_list
  for line in `cat $scp_list`; do

    wavname=`basename -s .scp $line`
    w=`echo $wavname | awk -F '_' 'BEGIN{OFS="_"}{print $1,$2,$3,$4}'` 
    w=${w}.wav
    copy-feats scp:$line ark,scp:tmp_enhan.ark,tmp_enhan.scp || exit 1
    python  ./utils/reconstruct_spectrogram.py tmp_enhan.scp ${ori_wav_path}/$w ${rec_wav_path}/${wavname} || exit 1
  done

rm tmp_enhan.* $scp_list
echo "Done OK!"
fi
