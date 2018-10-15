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


def removeClipping(currents, CV = 0):
    '''
    Removes clipped data from the measured data.
    
    Parameters:
        currents: output currents
        CV: used control voltages (optional)
    Returns:
        cleanCV: use _ if not used
        cleanCurrents: array of currents without the clipped samples
    '''
    
    Imean = np.mean(currents, axis = 1)
    cleanCurrents = currents[abs(Imean) < 3.1] # TODO: find exact clipping value
    cleanCV = 0
    if CV != 0:     # CV data is optional
        cleanCV = CV[abs(Imean) < 3.1]
    return cleanCV, cleanCurrents
    

def createPSD(data, fs, window = 'hann'):
    '''
    Computes the PSDs of a dataset of output currents
    
    Parameters:
        data: (samples, sample points)
        fs: sample frequency
        window: window for Welch's method (default: hann)
    '''
    m = data.shape[0]
    n = int(fs/2) + 1
    P = np.zeros((m, n))
    f = 0
    for i in range(m):
        [f,P[i,:]] = welch(data[i,:], fs, window, fs)
    if type(f) == int:
        print('test')
    return f, P

def PSDPlotter(f, P):
    """
    Plots the PSDs of the given samples in one plot.
    
    Parameters:
        f: freq list
        P: PSDs
    """
    for i in range(P.shape[0]):
        plt.plot(f,P[i,:])
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (I**2/Hz)')

def variancePSD(f, P, f_cut = None, window = 'hann'):
    '''
    Computes the variance of the signal up to a specific frequency given the PSDs.
    
    Parameters:
        f: list of frequencies corresponding to the PSD values
        P: PSD values
        f_cut: maximum frequency for variance calculation (optional)
    '''
    
    area = np.zeros((P.shape[0],1))
    cutoff = f[-2]
    if f_cut != None:
        cutoff = f_cut
    for j in range (P.shape[0]):
        i = 1
        area[j] = 0
        while cutoff >= f[i]:        
            area[j] += P[j, i - 1] * (f[i] - f[i - 1]) if P.shape[0] > 1 else P[i - 1] * (f[i] - f[i - 1]) 
            i += 1
    return area


def varianceSpread(currents):
    """
    Computes the variance of the current variance of a CV configuration.
    
    Parameters:
        currents: array of currents for the samples
    returns:
        var: variance of the currents
        spread: variance of the variance of different samples
    """
    
    _, currents = removeClipping(currents)
    Ivar = np.var(currents, axis = 1)
    spread = np.var(Ivar)
    return Ivar, spread


def spreadPlotter(currents, nr_samples, sampleSize):
    """
    Plots the spread of the variance found of the sampled data versus the mean.
    The variance is calculated using np.var(), not a PSD integral.
    
    Parameters:
        currents: measured currents
        nr_samples: amount of different CV configs
        sampleSize: amount of samples for one specific CV config
    Returns:
        Ivar: variance of the currents
        spread: variance of the variance of currents with the same CV config
    """
    _, currents = removeClipping(currents)
    spread = np.zeros((nr_samples,1))
    Imean = np.mean(currents, axis = 1)
    Ivar = np.var(currents, axis = 1)
    
    for i in range(nr_samples):
        spread[i] = np.var(Ivar[i * sampleSize : (i + 1) * sampleSize])
        spread[i] /= np.mean(Ivar[i * sampleSize : (i + 1) * sampleSize])
    
    plt.figure()
    plt.plot(Imean, Ivar,'.')
    plt.xlabel('Mean current')
    plt.ylabel('Variance of current')
    plt.title('Variance spread of CV configurations')
    plt.tight_layout()     
    return Ivar, spread

