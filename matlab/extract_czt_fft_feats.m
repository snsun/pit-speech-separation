%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This script is used to extract CZT+FFT features.
%% 1. We use the 128-point CZT to 50-1000Hz as additional features
%%    to improve the frequncy resolution.
%% 2. Our final feature is 128 + 129 = 257 dim;
%% 3. Note: I comment the part which is used to extract FFT features
%%    for target because we have had the feature. If you don't have
%%    targets features, please uncomment this part.

mode1 = {'tr'}; % in order to run parallelly, extract 'tr', 'cv' and 'tt' separately
mode_len = length(mode1);
data_dir = '../data/wav/wav8k/min/'; %CHANGE THE DIR TO YOUR DATA
feats_dir = '../data/feats/50_1000_128_zoomfft/feats_8k_czt_psm/';    %CHANGE THE DIR TO WHERE YOU WANT TO STORE THE FEATURES
for idx=1:mode_len
    mode = mode1{idx};
    input_dir = [data_dir  mode '/']; 
    mix_dir = [input_dir 'mix/'];
    s1_dir = [input_dir 's1/'];
    s2_dir = [input_dir 's2/'];
    
    output_feats_dir = [feats_dir mode '_inputs/'];
    output_labels_dir = [feats_dir  mode '_labels/'];
    mkdir(output_feats_dir);
    mkdir(output_labels_dir);
    
    
    fid_feats = fopen([output_feats_dir 'feats.txt'], 'w');
    fid_labels = fopen([output_labels_dir 'feats.txt'], 'w');
    
    % FFT and CZT configuration
    fs = 8000;
    fft_len = 256;
    dim = 129;
    frame_len = 256;
    frame_shift = 128;

    f1 = 50; %in Hz, CZT start freq
    f2 = 1000; %in Hz, CZT end freq
    M = 128; % CZT poits
    w=exp(-j*2*pi*(f2-f1)/(M*fs));% for CZT
    a=exp(j*2*pi*f1/fs);% for CZT

    Win=sqrt(hamming(fft_len,'periodic'));
    Win=Win/sqrt(sum(Win(1:frame_shift:fft_len).^2));
    lists = dir(mix_dir);
    for i = 3:length(lists)
        utt_id = lists(i).name;
        filename = [mix_dir utt_id];
        wav = audioread(filename);
        frames = enframe(wav, Win, frame_shift);
        X = fft(frames, fft_len, 2);
        theta_y=angle(X(:, 1:dim));
        Y = abs(X(:, 1:dim)); 
        
        %CZT

        Y_c = abs(czt(frames', M, w, a));
        Y_c = Y_c';
        feats = [Y_c, Y];

        writekaldifeatures(fid_feats, utt_id, feats);
        filename1 = [s1_dir utt_id];
        wav1 = audioread(filename1);
        frames = enframe(wav1, Win, frame_shift);
        X = fft(frames , fft_len, 2);
        Y = abs(X(:, 1:dim));
        theta_s = angle(X(:, 1:dim));
        Y = Y.*cos(theta_y-theta_s);
        writekaldifeatures(fid_labels, [utt_id(1:end-4) '_1.wav'], Y);
        filename2 = [s2_dir utt_id];
        wav1 = audioread(filename2);
        frames = enframe(wav1, Win, frame_shift);
        X = fft(frames , fft_len, 2);
        Y = abs(X(:, 1:dim));
         theta_s = angle(X(:, 1:dim));
        Y = Y.*cos(theta_y-theta_s);
        writekaldifeatures(fid_labels, [utt_id(1:end-4) '_2.wav'], Y);   
        if mod(i, 100) == 0
              i
        end
    end
    fclose(fid_feats);
    fclose(fid_labels);

end
