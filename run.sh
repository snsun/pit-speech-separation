#! /bin/bash

# This recipe is used to do NN-PIT (LSTM, DNN or  BLSTM)
# I am sorry that we use different tools to do different work in this recipe
#   1. Feature extraction: Matlab-> czt & fft features -> Kaldi ark(text)
#   2. Feature format transformation: Kaldi ark(txt) -> Kaldi ark(binary), scp -> Tensorflow records
#      We define the tf records format for our task, please see the codes for the details
#   3. Traing & Test model: Tensorflow

step=3
kaldi_feats_dir=/home/disk2/snsun/workspace/separation/data/feats/50_1000_128_zoomfft/feats_8k_czt_psm/
          #give the feature dir where you store your feats, it must includes {tr, cv, tt}_{inputs, labels} dirctories
copy_labels=1

lists_dir=./tmp/lists/ #lists_dir is used to store some necessary files lists
mkdir -p $lists_dir
apply_cmvn=1
num_threads=12
tfrecords_dir=data/tfrecords/50_500_64_zoomfft/
inputs_cmvn=$kaldi_feats_dir/tr_inputs/cmvn.ark
labels_cmvn=''

gpu_id='0'
TF_CPP_MIN_LOG_LEVEL=1
rnn_num_layers=2
tr_batch_size=32
tt_batch_size=1
input_size=257
output_size=129
rnn_size=128
keep_prob=0.8
learning_rate=0.0005
halving_factor=0.7
decode=0
model_type=BLSTM
prefix=ZoomFFT
assignment=opt
name=${prefix}_${model_type}_${rnn_num_layers}_${rnn_size}
save_dir=exp/$name/
data_dir=data/separated/${name}_${assignment}/
resume_training=false
czt_dim=128

#for step 5
#ori_wav_path=/home/disk1/snsun/Workspace/tensorflow/kaldi/data/wsj0/create-speaker-mixtures/data/2speakers/wav8k/min/tt/mix/
ori_wav_path=/home/disk1/jqliu/LSTM_PIT/data/wav/mix_tt_8k/
  #rec_wav_path=data/wav/rec_deepcluster_${fs}_${assign}/
rec_wav_path=data/wav/rec/${name}_${assignment}/

#Step 0: extract features using matlab program. 
#    Note: You need to change the data_dir path and kaldi_feats_dir path in 
#          matlab_feats_extraction/extract_czt_fft_feats.m  accordng to your config;

if [ $step -le 0 ]; then
    echo " please run the script 'matlab_feats_extraction/extract_czt_fft_feats.m' \n
           in matlab and tell me the feature folder path for the next steps!" && exit 1

fi

#####################################################################################################
#   NOTE for STEP 1:                                                                              ###
#       1.you need to check if you give the right 'kaldi_feats_dir' and 'copy_labels' in config session ###
#       2.make sure that your path.sh includes the right Kaldi path!!                             ###
#####################################################################################################
if [ $step -le 1 ] ; then
    echo " Feature format transformatio \n copy text ark to binary ark and scp and calculate the Mean and Variance for inputs." 

    . ./path.sh #This is the Kaldi path file
    for x in tr cv tt; do 
        if $copy_labels; then
            for y in inputs labels;do
                copy-feats ark:$kaldi_feats_dir/${x}_${y}/feats.txt ark,scp:$kaldi_feats_dir/${x}_${y}/feats.ark,$kaldi_feats_dir/${x}_${y}/feats.scp &
            done
        else
            for y in inputs; do
                copy-feats ark:$kaldi_feats_dir/${x}_${y}/feats.txt ark,scp:$kaldi_feats_dir/${x}_${y}/feats.ark,$kaldi_feats_dir/${x}_${y}/feats.scp &
            done
        fi
        compute-cmvn-stats ark:$kaldi_feats_dir/${x}_inputs/feats.txt $kaldi_feats_dir/${x}_${y}/cmvn.ark &
    done
    wait 
fi

#####################################################################################################
#   NOTE for STEP 2:                                                                              ###
#       1.you need to check the following config  in config session                               ###
#           lists_dir: used to store tfrecords file lists;                                        ###
#           apply_cmvn: 1 or 0, if you use cmvn when you transform the feature                    ###
#           num_threads: convert the data parallelly                                              ###
#           tfrecords_dir: where you want to store the tfrecords file                             ###
#           inputs_cmvn: the cmvn file for inputs computed using Kaldi                            ###
#           labels_cmvn: the cmvn file for outputs (just for regression task),                    ###        
#                        if you don't use cmvn for labels, make sure labels_cmvn=''               ###
#####################################################################################################

if [ $step -le 2 ] ; then
    echo "Transform the kaldi features to tf records"
    for mode in tt tr cv; do # generated list name is $lists_dir/$mode_feats_mapping.lst
        python local/makelists.py $kaldi_feats_dir  $mode $lists_dir
        python local/convert_to_records.py --mapping_list=$lists_dir/${mode}_feats_mapping.lst \
        --inputs_cmvn=$inputs_cmvn --labels_cmvn=$labels_cmvn --output_dir=$tfrecords_dir/$mode/ --num_threads=$num_threads\
        --apply_cmvn=$apply_cmvn & 

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
        find $tfrecords_dir/${x}/ -iname "*.tfrecords" > $lists_dir/${x}.lst
    done
    tr_cmd="python run_lstm.py \
    --lists_dir=$lists_dir  --rnn_num_layers=$rnn_num_layers --batch_size=$batch_size --rnn_size=$rnn_size \
    --decode=$decode --learning_rate=$learning_rate --save_dir=$save_dir --data_dir=$data_dir --keep_prob=$keep_prob \
    --input_size=$input_size --output_size=$output_size  --assign=$assignment --resume_training=$resume_training \
    --model_type=$model_type --halving_factor=$halving_factor --czt_dim=$czt_dim"

    echo $tr_cmd
    CUDA_VISIBLE_DEVICES=$gpu_id TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL $tr_cmd
fi

#####################################################################################################
#   NOTE for STEP 4:                                                                              ###
#       1. Make sure that you configure the RNN/data_dir/model_dir/ all rights                    ###
#####################################################################################################

if [ $step -le 4 ]; then
    
    echo "Start Decoding."
    decode=1
    batch_size=1
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





