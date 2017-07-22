addpath('../voicebox');
mode1 = {'tr', 'cv', 'tt'};
for idx=1:3
mode = mode1{idx};
input_dir = ['data/2speakers/wav8k/min/' mode '/'];
mix_dir = [input_dir 'change_pitch_mix/'];
s1_dir = [input_dir 's1/'];
s2_dir = [input_dir 's2/'];

output_feats_dir = ['feats_8k_change_pitch/' mode '_inputs/'];
output_labels_dir = ['feats_8k/' mode '_labels/'];
mkdir(output_feats_dir);
mkdir(output_labels_dir);


fid_feats = fopen([output_feats_dir 'feats.txt'], 'w');
fid_labels = fopen([output_labels_dir 'feats.txt'], 'w');
fft_len = 256;
dim = 129;
frame_len = 256;
frame_shift = 128;
W=sqrt(hamming(fft_len,'periodic'));
W=W/sqrt(sum(W(1:frame_shift:fft_len).^2))
lists = dir(mix_dir)
for i = 3:length(lists)
    utt_id = lists(i).name;
    filename = [mix_dir utt_id];
    wav = audioread(filename);
    frames = enframe(wav, W, frame_shift);
    X = fft(frames, fft_len, 2);
    Y = abs(X(:, 1:dim));
    writekaldifeatures(fid_feats, utt_id, Y);
%    filename1 = [s1_dir utt_id];
%    wav1 = audioread(filename1);
%    frames = enframe(wav1, W, frame_shift);
%    X = fft(frames , fft_len, 2);
%    Y = abs(X(:, 1:dim));
%    writekaldifeatures(fid_labels, [utt_id(1:end-4) '_1.wav'], Y);
%    filename2 = [s2_dir utt_id];
%    wav1 = audioread(filename2);
%    frames = enframe(wav1, W, frame_shift);
%    X = fft(frames , fft_len, 2);
%    Y = abs(X(:, 1:dim));
%    writekaldifeatures(fid_labels, [utt_id(1:end-4) '_2.wav'], Y);   
    if mod(i, 100) == 0
          i
    end
end
fclose(fid_feats);
fclose(fid_labels);

end
