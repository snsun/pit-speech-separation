#!/usr/bin/env python
#by wujian@2017.4.15

import wave
import numpy as np


class WaveWrapper(object):

    def __init__(self, path, time_wnd = 25, time_off = 10):
        wave_src = wave.open(path, "rb")
        para_src = wave_src.getparams()
        self.rate = int(para_src[2])
        self.cur_size = 0
        self.tot_size = int(para_src[3])
        # default 400 160
        self.wnd_size = int(self.rate * 0.001 * time_wnd)
        self.wnd_rate = int(self.rate * 0.001 * time_off)
        self.ham = np.hamming(self.wnd_size+1)
	self.ham = np.sqrt(self.ham[0:self.wnd_size])
	self.ham = self.ham / np.sqrt(np.sum(np.square(self.ham[range(0,self.wnd_size, self.wnd_rate)])))
        self.data = np.fromstring(wave_src.readframes(wave_src.getnframes()), dtype=np.int16)
        self.upper_bound = np.max(np.abs(self.data))

    def get_frames_num(self):
        return int((self.tot_size - self.wnd_size) / self.wnd_rate + 1)

    def get_wnd_size(self):
        return self.wnd_size

    def get_wnd_rate(self):
        return self.wnd_rate

    def get_sample_rate(self):
        return self.rate

    def get_upper_bound(self):
        return self.upper_bound

    def next_frame_phase(self,fft_len=512, pre_em=True):
        while self.cur_size + self.wnd_size <= self.tot_size:
            value = np.zeros(fft_len)
            value[: self.wnd_size] = np.array(self.data[self.cur_size: \
                    self.cur_size + self.wnd_size], dtype=np.float)
            if pre_em:
              value -= np.sum(value) / self.wnd_size
              value[1: ] -= value[: -1] * 0.97
              value[0] -= 0.97 * value[0]
            value[: self.wnd_size] *= self.ham
            angle = np.angle(np.fft.rfft(value))
            yield np.cos(angle) + np.sin(angle) * 1.0j
            self.cur_size += self.wnd_rate
