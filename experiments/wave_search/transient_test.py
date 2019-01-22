# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 12:14:28 2019

This script tests for transients in the sampled data. It requires the input data and the amount of test samples to be taken.

@author: Mark
"""
from SkyNEt.instruments import InstrumentImporter
import numpy as np
import time
import matplotlib.pyplot as plt

def transient_test(waves, data, fs, sampleTime, n):

    adwin = InstrumentImporter.adwinIO.initInstrument()
    testdata = np.zeros((n, 2*fs))
    test_cases = np.random.randint(waves.shape[1], size=(1,n)) # Index for the wave
    difference = np.zeros((n,1))
    
    for i in range(n):     
        start_wave = time.time()
        testdata[i,:] = InstrumentImporter.nidaqIO.IO_cDAQ(np.ones((waves.shape[0], 2*fs)) * waves[:, test_cases[0,i],np.newaxis], fs) # sample for 2s

        difference[i,0] = np.mean(testdata[i,int(0.5*fs):2*fs]) - data[0, test_cases[0,i]] # use only last 1.5 seconds of test data (to avoid transients)
        end_wave = time.time()
        print('Transient test data point ' + str(i+1) + ' of ' + str(n) + ' took ' + str(end_wave-start_wave)+' sec.')

    plt.plot(data[0])
    plt.plot(test_cases[0,:], np.mean(testdata, axis=1), '.')
    plt.show()

    return testdata, difference, test_cases
