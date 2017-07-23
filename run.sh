#! /bin/bash

# This recipe is used to do NN-PIT (LSTM, DNN or  BLSTM)
# I am sorry that we use different tools to do different work in this recipe
#   1. Feature extraction: Matlab-> czt & fft features -> Kaldi ark(text)
#   2. Feature format transformation: Kaldi ark(txt) -> Kaldi ark(binary), scp -> Tensorflow records
#      We define the tf records format for our task, please see the codes for the details
#   3. Traing & Test model: Tensorflow

step=2
#Step 0: extract features using matlab program. 
#    Note: You need to change the data_dir path and feats_dir path in 
#          matlab_feats_extraction/extract_czt_fft_feats.m  accordng to your config;

if [ $step -le 0 ]; then
    echo " please run the script 'matlab_feats_extraction/extract_czt_fft_feats.m' \n
           in matlab and tell me the feature folder path for the next steps!" && exit 1

fi

feats_dir=/home/disk1/snsun/Workspace/tensorflow/kaldi/data/wsj0/create-speaker-mixtures/feats_8k_czt/
          #give the feature dir where you store your feats, it must includes {tr, cv, tt}_{inputs, labels} dirctories
copy_labels=false

if [ $step -le 1 ] ; then
    echo " Feature format transformatio \n copy text ark to binary ark and scp and calculate the Mean and Variance for inputs." 

    source path.sh #This is the Kaldi path file
    for x in tr cv tt; do 
        if $copy_labels; then
            for y in inputs labels;do
                copy-feats ark:$feats_dir/${x}_${y}/feats.txt ark,scp:$feats_dir/${x}_${y}/feats.ark,$feats_dir/${x}_${y}/feats.scp &
            done
        else
            for y in inputs; do
                copy-feats ark:$feats_dir/${x}_${y}/feats.txt ark,scp:$feats_dir/${x}_${y}/feats.ark,$feats_dir/${x}_${y}/feats.scp &
            done
        fi
#        compute-cmvn-stats ark:$feats_dir/${x}_inputs/feats.txt $feats_dir/${x}_${y}/cmvn.ark &
    done
    wait 
fi

lists_dir=./lists/ #lists_dir is used to store some necessary files lists
apply_cmvn=1
num_threads=12
tfrecords_dir=/home/disk1/snsun/Workspace/tensorflow/kaldi/data/tfrecords/8k_czt/
inputs_cmvn=$feats_dir/tr_inputs/cmvn.ark

if [ $step -le 2 ] ; then
    echo "Transform the kaldi features to tf records"
    for mode in tt tr cv; do # generated list name is $lists_dir/$mode_feats_mapping.lst
        python utils/makelists.py $feats_dir  $mode $lists_dir
        python utils/convert_to_records.py --mapping_list=$lists_dir/${mode}_feats_mapping.lst \
        --inputs_cmvn=$inputs_cmvn --labels_cmvn='' --output_dir=$tfrecords_dir/$mode/ --num_threads=$num_threads\
        --apply_cmvn=$apply_cmvn & 

    done
    wait
fi

## For the training, we need proper tfrecords_dir to find the tfrecords features and labels
## For the test, we  also need the 'tt' mixed speech wav dir because we need to extract phase info when we reconstruct wav








