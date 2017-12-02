mixed_wav_dir = '/home/disk2/snsun/workspace/separation//data/wav/wav8k/min/tt/mix/';
spk1_dir = '/home/disk2/snsun/workspace/separation/data/wav/wav8k/min/tt/s1/';
spk2_dir = '/home/disk2/snsun/workspace/separation/data/wav/wav8k/min/tt//s2/';
model_name='StandPsmPIT_BLSTM_3_400_def';
rec_wav_dir = ['../data/separated/' model_name  '/'];
lists = dir(spk2_dir);
len = length(lists) - 2;
SDR =  zeros(len, 2);
SIR = SDR;
SAR = SDR;
SDR_Mix = SDR;
SIR_Mix = SDR;
SAR_Mix = SDR;
for i = 3:len+2
    name = lists(i).name;
    part_name = name(1:end-4);
    rec_wav1 = audioread([rec_wav_dir part_name '_1.wav']);
    rec_wav2 = audioread([rec_wav_dir part_name '_2.wav']);
    rec_wav = [rec_wav1,rec_wav2];
    
    ori_wav1 = audioread([spk1_dir part_name '.wav']);
    ori_wav2 = audioread([spk2_dir part_name '.wav']);
    ori_wav = [ori_wav1, ori_wav2];
    
    mix_wav1 = audioread([mixed_wav_dir part_name '.wav']);
    mix_wav = [mix_wav1, mix_wav1];
    
    min_len = min(size(ori_wav, 1), size(rec_wav, 1));
    rec_wav = rec_wav(1:min_len, :);
    ori_wav = ori_wav(1:min_len, :);
    mix_wav = mix_wav(1:min_len, :);
    [SDR(i-2, :),SIR(i-2, :),SAR(i-2, :),perm]=bss_eval_sources(rec_wav',ori_wav');
    if mod(i, 200) == 0
        i
    end
end
fprintf('The mean SDR is %f', mean(mean(SDR)))
save(['sdr_' model_name], 'SDR', 'SAR', 'SIR', 'lists');

% Calculte different gender case
[spk, gender] = textread('spk2gender', '%s%d');
cmm = 1;
cmf = 1;
cff = 1;
for i = 1:size(SDR, 1)
    mix_name = lists(i+2).name;
    spk1 = mix_name(1:3);
    tmp = regexp(mix_name, '_');
    spk2 = mix_name(tmp(2)+1:tmp(2)+3);
    for j = 1:length(spk)
        if spk1 == spk{j}
            break
        end
    end
    for k = 1:length(spk)
        if spk2 == spk{k}
            break
        end
    end
    
    if gender(k) == 0 & gender(j) == 0
        SDR_FF(cff,:) = SDR(i, :); 
        lists_FF{cff} = lists(i).name;
        cff = cff +1;
    
    elseif gender(k) == 1 & gender(j) == 1
        SDR_MM(cmm,: )= SDR(i, :); 
        lists_MM{cmm} = lists(i).name;
        cmm = cmm + 1;
    else
        SDR_MF(cmf, :) = SDR(i, :);
        lists_MF{cmf} = lists(i).name;
        cmf = cmf + 1;
    end
end
fprintf('The mean SDR for Male & Female is : %f', mean(mean(SDR_MF)));
fprintf('The mean SDR for Female & Female is : %f', mean(mean(SDR_FF)));
fprintf('The mean SDR for Male & Male is : %f', mean(mean(SDR_MM)));

