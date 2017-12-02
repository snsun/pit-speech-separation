model_name='ZoomFFT_BLSTM_3_496_10_26_def.mat'
load(['sdr_' model_name], 'SDR', 'SAR', 'SIR', 'lists');
fprintf('The mean SDR is %f', mean(mean(SDR)))

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

