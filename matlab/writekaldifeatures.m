function [fid] = writekaldifeatures(fid, utt_id, data)

% WRITEKALDIFEATURES Writes a set of features in Kaldi format
%
% writekaldifeatures(features,filename)
%
% Inputs:
% features: set of features in Matlab format (see readkaldifeatures for
% detailed format specification)
% filename: Kaldi feature filename (.ARK extension)
%
% Note: a .SCP file containing the location of the output .ARK file is also
% created
%
% If you use this software in a publication, please cite
% Emmanuel Vincent and Shinji Watanabe, Kaldi to Matlab conversion tools, 
% http://kaldi-to-matlab.gforge.inria.fr/, 2014.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2014 Emmanuel Vincent (Inria) and Shinji Watanabe (MERL)
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



feature=data;
fprintf(fid,'%s  [\n ', utt_id);
nfram=size(feature,1);
for t=1:nfram,
    fprintf(fid,' %.7g', feature(t, :));
    fprintf(fid,' \n ');
end
fprintf(fid,' ]\n');

return