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
    rampT = int(fs/2)
    iterations = 1 # iterate multiple times over the test cases
    testdata = np.zeros((iterations, n, (T*fs - 2*rampT)))
    #testdata = np.zeros((n, T*fs))
    test_cases = np.random.randint(waves.shape[1], size=(1,n)) # Index for the wave
    test_cases = np.sort(test_cases)
    difference = np.zeros((iterations, n, 1))
    
    for it in range(iterations):
        for i in range(n):     
            start_wave = time.time()

            wavesRamped = np.zeros((waves.shape[0], T*fs)) # 1.5 second to ramp up to desired input, T-3 seconds measuring, 1.5 second to ramp input down to zero
            dataRamped = np.zeros(wavesRamped.shape[1])
            for j in range(wavesRamped.shape[0]):
                # Ramp up linearly (starting from value CV/2 since at CV=0V nothing happens anyway) and ramp back down to 0
                wavesRamped[j,0:rampT] = np.linspace(0,waves[j,test_cases[0,i]], rampT) 
                wavesRamped[j,rampT: rampT + (T*fs-2*rampT)] = np.ones((T*fs-2*rampT)) * waves[j, test_cases[0,i],np.newaxis]
                wavesRamped[j,rampT + (T*fs-2*rampT):] = np.linspace(waves[j,test_cases[0,i]], 0, rampT)

            dataRamped = InstrumentImporter.nidaqIO.IO_cDAQ(wavesRamped, fs)
            testdata[it, i, :] = dataRamped[0,rampT: rampT + (T*fs-2*rampT)]
            #testdata[i,:] = dataRamped[0, :]

            difference[it, i, 0] = np.mean(testdata[it, i,:]) - data[test_cases[0,i]] 
            end_wave = time.time()
            print('Transient test data point ' + str(it*n + i+1) + ' of ' + str(iterations * n) + ' took ' + str(end_wave-start_wave)+' sec.')

    #plt.plot(data)
    #plt.errorbar(test_cases[0,:], np.mean(testdata[-1,:,:], axis=1), yerr=np.amax(testdata[-1,:,:],axis=1) - np.amin(testdata[-1,:,:],axis=1),ls='',marker='o',color='r',linewidth=2) 
    #plt.plot(test_cases[0,:], np.mean(testdata, axis=1), '.')
    #plt.show()

    return testdata, difference, test_cases
