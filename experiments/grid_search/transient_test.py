# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 12:14:28 2019

This script tests for transients in the sampled data. It requires the input data,
the amount of test samples to be taken and the device that is used (nidaq or adwin).

@author: Mark
"""
from SkyNEt.instruments import InstrumentImporter
import nunmpy as np
import time

def transient_test(ivvi, voltages, waves, fs, n, device):

    adwin = InstrumentImporter.adwinIO.initInstrument()
    testdata = np.zeros((n, 3*fs))
    test_cases = np.random.randint(voltages.shape[0], size=n)
    difference = np.zeros((n,1))
    
    for i in range(n):     
        InstrumentImporter.IVVIrack.setControlVoltages(ivvi, voltages[test_cases[i], :])    
        time.sleep(1)
        if device == 'nidaq':
            testdata[i,:] = InstrumentImporter.nidaqIO.IO(np.ones((waves.shape[0], 3*fs)) * waves[:, fs * test_cases[i],np.newaxis], fs) # sample for 3s
        elif device == 'adwin':
        	testdata[i,:] = InstrumentImporter.adwinIO.IO(adwin, np.ones((waves.shape[0], 3*fs)) * waves[:, fs * test_cases[i],np.newaxis], fs)
        difference[i,0] = np.mean(testdata[i,fs:3*fs]) - data[0,fs * test_cases[i]] # use only last 2 seconds of test data (to avoid transients)
        
    return testdata, difference, test_cases
