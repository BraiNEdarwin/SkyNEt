# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 13:28:16 2018
Script containing all processing methods used for noise measurements.
@author: Mark Boon
"""
# SkyNet imports


# Other imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


'''
Removes clipped data from the measured data.
CV: used control voltages
currents: output currents
'''
def removeClipping(currents, CV = 0):
    Imean = np.mean(currents, axis = 1)
    cleanCurrents = currents[abs(Imean) < 3.1] # TODO: find exact clipping value
    cleanCV = 0
    if CV != 0:     # CV data is optional
        cleanCV = CV[abs(Imean) < 3.1]
    return cleanCV, cleanCurrents
    
'''
Computes the PSDs of a dataset of output currents
data: (samples, sample points)
fs: sample frequency
'''
def createPSD(data, fs, window = 'hann'):
    m = data.shape[0]
    n = int(fs/2) + 1
    P = np.zeros((m, n))
    f = 0
    for i in range(m):
        [f,P[i,:]] = welch(data[i,:], fs, window, fs)
    if type(f) == int:
        print('test')
    return f, P

'''
Computes the variance of the signal up to a specific frequency given the PSDs
f: list of frequencies corresponding to the PSD values
P: PSD values
f_cut: maximum frequency for variance calculation (optional)
'''
def variancePSD(f, P, f_cut = None, window = 'hann'):
    #[f, P] = createPSD(data, fs, window = 'hann')
    area = np.zeros((P.shape[0],1))
    cutoff = f[-2]
    if f_cut != None:
        cutoff = f_cut
    for j in range (P.shape[0]):
        i = 1
        area[j] = 0
        while cutoff >= f[i]:        
            area[j] += P[j, i - 1] * (f[i] - f[i - 1])
            i += 1
    return area


'''
Computes the variance of the current variance of a CV configuration.

returns:
    var: variance of the currents
    spread: variance of the variance of different samples
'''
def varianceSpread(currents, fs, f_cut = None, window = 'hann'):
    _, currents = removeClipping(currents)
    [f, P] = createPSD(currents, fs, window)
    var = variancePSD(f, P, f_cut)
    spread = np.var(var)
    return var, spread


'''
Plots the spread of the variance found of the sampled data.
currents: measured currents
nr_samples: amount of different CV configs
sampleSize: amount of samples for one specific CV config.
fs: sample frequency
'''
def spreadPlotter(currents, nr_samples, sampleSize, fs, f_cut = None, window = 'hann'):
    _, currents = removeClipping(currents)
    spread = np.zeros((nr_samples,1))
    Imean = np.mean(currents, axis = 1)
    [f,P] = createPSD(currents, fs, window)
    varPSD = variancePSD(f, P, f_cut, window)
    
    for i in range(nr_samples):
        spread[i] = np.var(varPSD[i * sampleSize : (i + 1) * sampleSize])
        spread[i] /= np.mean(varPSD[i * sampleSize : (i + 1) * sampleSize])
    
    plt.figure()
    plt.plot(Imean, varPSD,'.')
    plt.xlabel('Mean current')
    plt.ylabel('Variance of current')
    plt.title('Variance spread of CV configurations')
    plt.tight_layout()
        
    return varPSD, spread