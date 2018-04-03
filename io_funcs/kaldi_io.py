#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    
# Ke Wang  
# Sining Sun
# Yuchao Zhang 

"""IO classes for reading and writing kaldi .ark

This module provides io interfaces for reading and writing kaldi .ark files.
Currently, this module only supports binary-formatted .ark files. Text .ark
files are not supported.

To use this module, you need to provide kaldi .scp files only. The .ark
locations with corresponding offsets can be retrieved from .scp files.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import struct
import random
import numpy as np

class GlobalHeader(object):
    """ Compress ark format header. """
    def __init__(self, format, header):
        self.format = format
        self.min_value = header[0]
        self.range = header[1]
        self.num_rows = header[2]
        self.num_cols = header[3]

class PerColHeader(object):
    """ Compress ark format per column header. """
    def __init__(self, header):
        self.percentile_0 = header[0]
        self.percentile_25 = header[1]
        self.percentile_75 = header[2]
        self.percentile_100 = header[3]

class ArkReader(object):
    """ Class to read Kaldi ark format.

    Each time, it reads one line of the .scp file and reads in the
    corresponding features into a numpy matrix. It only supports
    binary-formatted .ark files. Text files are not supported.

    Attributes:
        utt_ids: A list saving utterance identities.
        scp_data: A list saving .ark path and offset for items in utt_ids.
        scp_position: An integer indicating which utt_id and correspoding
            scp_data will be read next.
    """

    def __init__(self, scp_path):
        """Init utt_ids along with scp_data according to .scp file."""
        self.scp_position = 0
        fin = open(scp_path,"r")
        self.utt_ids = []
        self.scp_data = []
        line = fin.readline()
        while line != '' and line != None:
            utt_id, path_pos = line.replace('\n','').split(' ')
            path, pos = path_pos.split(':')
            self.utt_ids.append(utt_id)
            self.scp_data.append((path, pos))
            line = fin.readline()

        fin.close()

    def shuffle(self):
        """Shuffle utt_ids along with scp_data and reset scp_position."""
        zipped = zip(self.utt_ids, self.scp_data)
        random.shuffle(zipped)
        self.utt_ids, self.scp_data = zip(*zipped)  # unzip and assign
        self.scp_position = 0

    @staticmethod
    def read_ark(self, ark_file, ark_offset=0):
        """Read data from the archive (.ark from kaldi).

        Returns:
            A numpy matrix containing data of ark_file.
        """
        ark_read_buffer = open(ark_file, 'rb')
        ark_read_buffer.seek(int(ark_offset), 0)
        header = struct.unpack('<xcccc', ark_read_buffer.read(5))
        if header[0] != "B":
            print(ark_file)
            print("Input .ark file is not binary")
            sys.exit(1)
        if header[1] == "C":
            if header[2] == "M" and header[3] != "2":
                header_read = struct.unpack('<ffii', ark_read_buffer.read(16))
                if header_read[3] == 0:
                    print("Empty matrix.")
                    sys.exit(1)
                else:
                    global_header = GlobalHeader(1, header_read)
                    ark_mat = self.read_compress(self, global_header, ark_read_buffer)
                return ark_mat
            else:
                print("Unsupport format.")
                print("Maybe because of the matrices with 8 or fewer rows.")
                sys.exit(1)
        else:
             _, rows = struct.unpack('<bi', ark_read_buffer.read(5))
             _, cols = struct.unpack('<bi', ark_read_buffer.read(5))
             if header[1] == "F":
                 tmp_mat = np.frombuffer(ark_read_buffer.read(rows * cols * 4),
                             dtype = np.float32)
             elif header[1] == "D":
                 tmp_mat = np.frombuffer(ark_read_buffer.read(rows * cols * 8),
                             dtype = np.float64)
             ark_mat = np.reshape(tmp_mat, (rows, cols))
             return ark_mat
        return None

    @staticmethod
    def uint16_to_float(global_header, value):
        """ Convert uint16 to float. """
        # the constant 1.52590218966964e-05 is 1/65535.
        return global_header.min_value \
            + global_header.range * 1.52590218966964e-05 * value

    @staticmethod
    def char_to_float(p0, p25, p75, p100, value):
        """ Convert char to float. """
        if value < 64 :
            return p0 + (p25 - p0) * value * (1/64.0)
        elif value <= 192:
            return p25 + (p75 - p25) * (value - 64) * (1/128.0)
        else:
            return p75 + (p100 - p75) * (value - 192) * (1/63.0)

    @staticmethod
    def read_compress(self, header, buf):
        """Read compress ark format.

        Args:
        header: GlobalHeader format data.
        buf: PerColHeader and the byte data.
        """
        col_header = []
        ark_mat = np.zeros((header.num_rows, header.num_cols))

        for i in range(header.num_cols):
            col_header.append(PerColHeader(\
                    struct.unpack('<HHHH', buf.read(8))))

        for i in range(header.num_cols):
            p0 = self.uint16_to_float(header, col_header[i].percentile_0)
            p25 = self.uint16_to_float(header, col_header[i].percentile_25)
            p75 = self.uint16_to_float(header, col_header[i].percentile_75)
            p100 = self.uint16_to_float(header, col_header[i].percentile_100)
            for j in range(header.num_rows):
                value = struct.unpack('<B', buf.read(1))[0]
                ark_mat[j, i] = self.char_to_float(p0, p25, p75, p100, value)
        return ark_mat

    def read_next_utt(self):
        """Read the next utterance in the scp file.

        Returns:
            The utterance ID of the utterance that was read, the utterance
            data, and a bool indicating if the reader looped back to the
            beginning.
        """
        if len(self.scp_data) == 0:
             return None , None, True

        if self.scp_position >= len(self.scp_data):  #if at end of file loop around
            looped = True
            self.scp_position = 0
        else:
            looped = False

        self.scp_position += 1

        utt_ids = self.utt_ids[self.scp_position-1]
        utt_data = self.read_utt_data_from_index(self.scp_position-1)

        return utt_ids, utt_data, looped

    def read_next_scp(self):
        """Read the next utterance ID but don't read the data.

        Returns:
            The utterance ID of the utterance that was read.
        """
        if self.scp_position >= len(self.scp_data):  #if at end of file loop around
            self.scp_position = 0

        self.scp_position += 1

        return self.utt_ids[self.scp_position-1]

    def read_previous_scp(self):
        """Read the previous utterance ID but don't read the data.

        Returns:
            The utterance ID of the utterance that was read.
        """
        if self.scp_position < 0:  #if at beginning of file loop around
            self.scp_position = len(self.scp_data) - 1

        self.scp_position -= 1

        return self.utt_ids[self.scp_position+1]

    def read_utt_data_from_id(self, utt_id):
        """Read the data of a certain utterance ID.

        Args:
            utt_id: A string indicating a certain utterance ID.

        Returns:
            A numpy array containing the utterance data corresponding to the ID.
        """
        utt_mat = self.read_utt_data_from_index(self.utt_ids.index(utt_id))

        return utt_mat

    def read_utt_data_from_index(self, index):
        """Read the data of a certain index.

        Args:
            index: A integer index corresponding to a certain utterance ID.

        Returns:
            A numpy array containing the utterance data corresponding to the
            index.
        """
        return self.read_ark(self, self.scp_data[index][0], self.scp_data[index][1])

    def split(self):
        """Split of the data that was read so far."""
        self.scp_data = self.scp_data[self.scp_position:-1]
        self.utt_ids = self.utt_ids[self.scp_position:-1]


class ArkWriter(object):
    """Class to write numpy matrices into Kaldi .ark file and create the
    corresponding .scp file. It only supports binary-formatted .ark files.
    Text and compressed .ark files are not supported.

    Attributes:
        scp_path: The path to the .scp file that will be written.
        scp_file_write: The file object corresponds to scp_path.

    """

    def __init__(self, scp_path):
        """Arkwriter constructor."""
        self.scp_path = scp_path
        self.scp_file_write = open(self.scp_path, "w")

    def write_next_utt(self, ark_path, utt_id, utt_mat):
        """Read an utterance to the archive.

        Args:
            ark_path: Path to the .ark file that will be used for writing.
            utt_id: The utterance ID.
            utt_mat: A numpy array containing the utterance data.
        """
        ark_file_write = open(ark_path,"ab")
        utt_mat = np.asarray(utt_mat, dtype=np.float32)
        rows, cols = utt_mat.shape
        ark_file_write.write(struct.pack('<%ds'%(len(utt_id)), utt_id))
        pos = ark_file_write.tell()
        ark_file_write.write(struct.pack('<xcccc','B','F','M',' '))
        ark_file_write.write(struct.pack('<bi', 4, rows))
        ark_file_write.write(struct.pack('<bi', 4, cols))
        ark_file_write.write(utt_mat)
        self.scp_file_write.write('%s %s:%s\n' % (utt_id, ark_path, pos))
        ark_file_write.close()

    def close(self):
        """close the ark writer"""
        self.scp_file_write.close()
