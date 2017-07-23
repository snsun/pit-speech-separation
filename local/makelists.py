# This python script is used to generate a file list for {tr, tt, cv}.
# In order to transform the Kaldi feats to tfrecords, we need a list to 
# specify the input and target kaldi file.Every list has the following form:
#  utt_id inputs_ark target1_ark target2_ark
# Usage:
# python makelists.py feats_dir mode list_dir
#   feats_dir: kaldi feats.ark, scp dir
#   mode: tr, cv, or tt
#   list_dir: where to store the generated list

import sys
import os

usage = '''
  Usage:
   python makelists.py feats_dir mode list_dir
      feats_dir: kaldi feats.ark, scp dir
      mode: tr, cv, or tt
      list_dir: where to store the generated list
'''
if len(sys.argv) is not 4:
    print usage;
    exit();

feats_dir = sys.argv[1];
mode = sys.argv[2]
list_dir = sys.argv[3]
if not os.path.exists(list_dir):
    os.makedirs(list_dir)

inputscp = feats_dir + '/' + mode + '_inputs/feats.scp'
outputscp = feats_dir +'/' + mode + '_labels/feats.scp'
lst=list_dir + '/' + mode + '_feats_mapping.lst'
fid1 = open(inputscp, 'r')
lines1 = fid1.readlines()
fid2 = open(outputscp, 'r')
lines2 = fid2.readlines()

fid1.close()
fid2.close()

fid3 = open(lst, 'w')

dict1 = {}
dict2 = {}
for line in lines2:
  l = line.rstrip('\n')
  strs = l.split(' ')
  dict1[strs[0]] = strs[1]

for line in lines1:
  line = line.rstrip('\n')
  strs = line.split(' ')
  utt = strs[0]
  cont = strs[1]
  (names, ext) = os.path.splitext(utt)
 
  name1 = names +'_1.wav'
  name2 = names + '_2.wav'
  fid3.write(utt + ' ' + cont +' ' + dict1[name1] + ' ' + dict1[name2] + '\n')

fid3.close()
  



 
