#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:01:04 2019

@author: annefleur
"""


import numpy as np

def GenWaveform(amplitudes, lengths, slopes=None):
    '''Generates a waveform with constant intervals of value amplitudes[i] for interval i of length[i]. The slopes argument
    is the number of points of the slope.
    '''
    wave = []
    if len(amplitudes)==len(lengths):  
        for i in range(len(amplitudes)):
            wave += [amplitudes[i]]*lengths[i]
            if (slopes is not None) and (i < (len(amplitudes)-1)):
                try:
                    wave += np.linspace(amplitudes[i],amplitudes[i+1],slopes[i]).tolist()
                except:
                    wave += np.linspace(amplitudes[i],amplitudes[i+1],slopes[0]).tolist()
            
    elif len(lengths)==1:
        #ramp up start measurement
        wave += np.linspace(0,amplitudes[0],slopes[0]).tolist()
        for i in range(len(amplitudes)):
            wave += [amplitudes[i]]*lengths[0]
            if (slopes is not None) and (i < (len(amplitudes)-1)):
                assert len(slopes) == 1, 'slopes argument must have length 1 since len(lengths)=1'
                wave += np.linspace(amplitudes[i],amplitudes[i+1],slopes[0]).tolist()
        #ramp down end measurement
        wave += np.linspace(amplitudes[-1],0,slopes[0]).tolist()
    else:
        assert 0==1, 'Assignment of amplitudes and lengths is not unique!'
  
    return wave

if __name__=='__main__':
    
    from matplotlib import pyplot as plt
    amplitudes, lengths = [3,1,-1,1,-1], [100]
    wave = GenWaveform(amplitudes, lengths, slopes = [20])
    print(len(wave))
    plt.figure()
    plt.plot(wave)
    plt.show()