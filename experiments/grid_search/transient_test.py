# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 12:14:28 2019

This script tests for transients in the sampled data. It requires the input data,
the amount of test samples to be taken and the device that is used (nidaq or adwin).

@author: Mark
"""
from SkyNEt.instruments import InstrumentImporter
import numpy as np
import time
import matplotlib.pyplot as plt

def transient_test(ivvi, voltages, waves, data, fs, sampleTime, n, device):

    adwin = InstrumentImporter.adwinIO.initInstrument()
    testdata = np.zeros((n, 2*fs))
    test_cases = np.random.randint(voltages.shape[0], size=(2,n))
    test_cases[1,:] = np.random.randint(waves.shape[1], size=n) # First row is index for the grid, second row is index for the wave
    difference = np.zeros((n,1))
    
    for i in range(n):     
        InstrumentImporter.IVVIrack.setControlVoltages(ivvi, voltages[test_cases[0,i], :])    
        time.sleep(1)
        if device == 'nidaq':
            testdata[i,:] = InstrumentImporter.nidaqIO.IO(np.ones((waves.shape[0], 2*fs)) * waves[:, test_cases[1,i],np.newaxis], fs) # sample for 2s
        elif device == 'adwin':
        	testdata[i,:] = InstrumentImporter.adwinIO.IO(adwin, np.ones((waves.shape[0], 2*fs)) * waves[:, test_cases[1,i],np.newaxis], fs)
        difference[i,0] = np.mean(testdata[i,int(0.5*fs):2*fs]) - data[0, sampleTime * fs * test_cases[0,i] + test_cases[1,i]] # use only last 1.5 seconds of test data (to avoid transients)
    
    plt.plot(data[0])
    plt.plot(test_cases[1,:], np.mean(testdata, axis=1), '.')
    plt.show()

    return testdata, difference, test_cases
