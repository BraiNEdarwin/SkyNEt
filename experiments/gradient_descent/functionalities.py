# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 17:52:24 2020

File containing GD problems other than Boolean logic.

@author: Mark
"""

import numpy as np
from SkyNEt.modules.GenWaveform import GenWaveform


def featureExtractor(feature, signallength, edgelength=0.01, fs=1000):
    '''
    Creates inputs and targets for the feature extractor task.
    16 features are made: 0000, 0001, ..., 1111.
    
    inputs:
        feature:        which of the 16 features is being learned (number 0 to 15).
        signallength:   length of measuring a single feature.
        edgelength:     
    
    outputs:
        t:  time signal
        x:  [4 x many] inputs
        W:  [many] weights (1 for inputs, 0 for edges)
        
    '''
    assert feature >= 0, "Feature number must be 0 or higher"
    assert feature < 16, "Feature number must be 15 or lower"
    
    signallength = signallength/16 # since Boolean logic works with total signal length, we divide by 16 for a single feature input
    
    samples = 16 * round(fs * signallength) + 15 * round(fs * edgelength)
    t = np.linspace(0, samples/fs, samples)
    x = np.zeros((4, samples))
    W = np.ones(samples,dtype=bool)
    target = np.zeros(samples)
    
    x[0] = np.asarray(GenWaveform([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],[round(fs * signallength)],[round(fs * edgelength)]))
    x[1] = np.asarray(GenWaveform([0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],[round(fs * signallength)],[round(fs * edgelength)]))
    x[2] = np.asarray(GenWaveform([0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],[round(fs * signallength)],[round(fs * edgelength)]))
    x[3] = np.asarray(GenWaveform([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],[round(fs * signallength)],[round(fs * edgelength)]))
    
    edges = np.equal(x[0] != 0, x[0] != 1)    
    W[edges] = 0 
    
    target[round(fs*(feature*signallength + max(0,(feature-1))*edgelength)):round(fs*((feature+1)*signallength + (feature)*edgelength))] = 1
    
    return t, x, W, target

    
    
