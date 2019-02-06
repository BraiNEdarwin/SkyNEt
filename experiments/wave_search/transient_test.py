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
    T = 4 # Amount of time of sampling one datapoint
    testdata = np.zeros((n, (T-1)*fs))
    test_cases = np.random.randint(waves.shape[1], size=(1,n)) # Index for the wave
    difference = np.zeros((n,1))
    
    for i in range(n):     
        start_wave = time.time()

        wavesRamped = np.zeros((waves.shape[0], T*fs)) # .5 second to ramp up to desired input, T-1 seconds measuring, 0.5 second to ramp input down to zero
        dataRamped = np.zeros((1,wavesRamped.shape[1]))
        for j in range(wavesRamped.shape[0]):
            wavesRamped[j,0:int(fs/2)] = np.linspace(0,waves[j,test_cases[0,i]], int(fs/2))
            wavesRamped[j,int(fs/2): int(fs/2) + (T-1)*fs] = np.ones((T-1)*fs) * waves[j, test_cases[0,i],np.newaxis]
            wavesRamped[j,int(fs/2) + (T-1)*fs:] = np.linspace(waves[j,test_cases[0,i]], 0, int(fs/2))

        dataRamped = InstrumentImporter.nidaqIO.IO(wavesRamped, fs)
        testdata[i,:] = dataRamped[0, int(fs/2): int(fs/2) + (T-1)*fs]
        #testdata[i,:] = InstrumentImporter.nidaqIO.IO_cDAQ(np.ones((waves.shape[0], 2*fs)) * waves[:, test_cases[0,i],np.newaxis], fs) # sample for 2s

        difference[i,0] = np.mean(testdata[i,:]) - data[test_cases[0,i]] 
        end_wave = time.time()
        print('Transient test data point ' + str(i+1) + ' of ' + str(n) + ' took ' + str(end_wave-start_wave)+' sec.')

    plt.plot(data)
    plt.errorbar(test_cases[0,:], np.mean(testdata, axis=1), yerr=np.amax(testdata,axis=1) - np.amin(testdata,axis=1),ls='',marker='o',color='r',linewidth=2) 
    #plt.plot(test_cases[0,:], np.mean(testdata, axis=1), '.')
    plt.show()

    return testdata, difference, test_cases
