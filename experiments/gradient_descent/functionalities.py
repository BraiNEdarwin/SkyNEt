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
        signallength:   length of total signal.
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
    edges = np.zeros(samples,dtype=bool)
    
    x[0] = np.asarray(GenWaveform([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],[round(fs * signallength)],[round(fs * edgelength)]))
    x[1] = np.asarray(GenWaveform([0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],[round(fs * signallength)],[round(fs * edgelength)]))
    x[2] = np.asarray(GenWaveform([0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],[round(fs * signallength)],[round(fs * edgelength)]))
    x[3] = np.asarray(GenWaveform([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],[round(fs * signallength)],[round(fs * edgelength)]))
    
    for i in range(1,16):
        edges[int(fs*(i*signallength + (i-1)*edgelength)):int(fs*((i)*signallength + i*edgelength))] = True  
    W[edges] = 0 
    
    target[round(fs*(feature*signallength + max(0,(feature-1))*edgelength)):round(fs*((feature+1)*signallength + (feature)*edgelength))] = 1
    
    return t, x, W, target


def booleanLogic(gate, signallength, edgelength=0.01, fs=1000):
    '''
    inputs:
        gate:           string containing 'AND', 'OR', ...
        signallength:   length of total signal.
        edgelength:     ramp time (in s) between input cases
    
    '''    
    signallength = signallength/4
    samples = 4 * round(fs * signallength) + 3 * round(fs * edgelength)
    t = np.linspace(0, samples/fs, samples)
    x = np.zeros((2, samples))
    W = np.ones(samples,dtype=bool)
    target = np.zeros(samples)
    edges = np.zeros(samples,dtype=bool)
    
    x[0] = np.asarray(GenWaveform([0,0,1,1],[round(fs * signallength)],[round(fs * edgelength)]))
    x[1] = np.asarray(GenWaveform([0,1,0,1],[round(fs * signallength)],[round(fs * edgelength)]))
    
    for i in range(1,4):
        edges[int(fs*(i*signallength + (i-1)*edgelength)):int(fs*((i)*signallength + i*edgelength))] = True
    
    W[edges] = 0 
    
    if gate == 'AND':
        target = np.asarray(GenWaveform([0,0,0,1],[round(fs * signallength)],[round(fs * edgelength)]))
    elif gate == 'OR':
        target = np.asarray(GenWaveform([0,1,1,1],[round(fs * signallength)],[round(fs * edgelength)]))
    elif gate == 'NAND':
        target = np.asarray(GenWaveform([1,1,1,0],[round(fs * signallength)],[round(fs * edgelength)]))
    elif gate == 'NOR':
        target = np.asarray(GenWaveform([1,0,0,0],[round(fs * signallength)],[round(fs * edgelength)]))
    elif gate == 'XOR':
        target = np.asarray(GenWaveform([0,1,1,0],[round(fs * signallength)],[round(fs * edgelength)]))
    elif gate == 'XNOR':
        target = np.asarray(GenWaveform([1,0,0,1],[round(fs * signallength)],[round(fs * edgelength)]))
    else:
        assert False, "Target gate not specified correctly."
    
    return t, x, W, target
