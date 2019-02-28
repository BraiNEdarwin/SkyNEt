# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:25:31 2019

@author: Mark
"""

import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

dims = 7
Fs = 1000
##############
# Train data #
##############
skip_points = 12 # During training not all sampled data is used. This parameter determines how many data points would be skipped.
factor = 0.1
trainset_fraction = 0.05 # Fraction of the train set used (0.5 means only use first half of sampled data)
freq2 = np.array([2,np.pi,5,7,13,17,19])
freq = factor*np.sqrt(freq2[:dims])
sampleTime = int(24*3600 * trainset_fraction)
Vmax = 0.9

t = np.arange(0, sampleTime, skip_points/Fs)

inputs = freq[:,np.newaxis]*t[np.newaxis]
phase = np.zeros((dims,1))
inputs = np.sin(2*np.pi*inputs+phase)*Vmax

#############
# Test data #
#############
skip_pointsT = 10
factorT = 0.087
freq2T = np.array([2,np.pi,5,7,13,17,19])
freqT = factorT*np.sqrt(freq2T[:dims])
sampleTimeT = 2*3600
inputsT = np.ones((dims,1))
Vmax = 0.9

tT = np.arange(0, sampleTimeT, skip_pointsT/Fs)
input_partT = freqT[:,np.newaxis]*tT[np.newaxis]
phaseT = np.ones((dims,1))
input_partT = np.sin(2*np.pi*input_partT+phaseT)*Vmax
inputsT = np.concatenate((inputsT, input_partT),axis=1)
inputsT = inputsT[:,1:]

# Determine the smallest distance to the sampled area w.r.t. the test data
radius = 0.15

def distanceToTestdata(gridpoints, inputs, testdata, radius):
    start_block = time.time()
    indices = np.arange(0,testdata.shape[1])
    np.random.shuffle(indices)
    testdata = testdata[:,indices] # [dims, gridpoints] dataset
    min_dist = np.zeros((1, gridpoints))
    empty_counter = 0
    for i in range(gridpoints): 
        filtered_inputs = inputs[:, abs(inputs[0,:] - testdata[0,i,np.newaxis]) <= radius]
        for j in range(1,dims):
            filtered_inputs = filtered_inputs[:, abs(filtered_inputs[j,:] - testdata[j,i,np.newaxis]) <= radius] 
            
        if filtered_inputs.size == 0:
            empty_counter +=1
        else:                    
            dist = np.linalg.norm(filtered_inputs - testdata[:,i,np.newaxis], axis = 0)
            min_dist[0,i] = dist.min()    
        #dist = np.linalg.norm(inputs - grid[:, i, j, k, l, np.newaxis], axis = 0)
        #min_dist[i,j,k,l] = dist.min()    
                    
    end_block = time.time()
    print("time elapsed: " + str(end_block - start_block))
    return min_dist, empty_counter, indices


gridpoints = 10000
#min_dist_brute, empty_counter_brute = distance5D_brute(grid, inputs)
min_dist, empty_counter, indices = distanceToTestdata(gridpoints, inputs, inputsT, radius = 0.5)

plt.figure()
#plt.hist(np.reshape(min_dist_brute, (min_dist_brute.size,1)),bins=50, normed = True,label = "Full grid search")
plt.hist(min_dist[0,:],bins=50, normed=True, label = "Random sample, n = " + str(gridpoints))
#plt.legend()
plt.title(str(dims) +"D, train set " + str(int(sampleTime/3600)) + " hours, test set " + str(int(sampleTimeT/3600)) + " hours")

