# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 13:28:16 2018
Script containing all processing methods used for noise measurements.
@author: Mark Boon
"""
# SkyNet imports


# Other imports
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy
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
    cleanCurrents = currents[abs(Imean) < 36] # TODO: find exact clipping value
    cleanCV = 0
    if CV != 0:     # CV data is optional
        cleanCV = CV[abs(Imean) < 3.1]
    return cleanCV, cleanCurrents
    
def currentPlotter(data):
    '''
    Plots the current in a single graph
    
    Parameters:
        data: the currents in an array
    '''
    x = np.linspace(0,data.shape[1]/1000, data.shape[1])
    for i in range(data.shape[0]):
        plt.plot(x, data[i,:])
    plt.xlabel('time')
    plt.ylabel('current')
    

def createPSD(data, fs, window = 'hann', nperseg = 256):
    '''
    Computes the PSDs of a dataset of output currents
    
    Parameters:
        data: (samples, sample points)
        fs: sample frequency
        window: window for Welch's method (default: hann)
        nperseg: Length of each segment
    '''
    m = data.shape[0]
    n = int(nperseg/2) + 1
    P = np.zeros((m, n))
    f = 0
    for i in range(m):
        [f,P[i,:]] = welch(data[i,:], fs, window, nperseg)
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
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (I**2/Hz)')

def noiseFit(f, P, frange, plot = True):
    """
    Finds the parameter gamma assuming b/f**gamma noise and plots the PSD with 
    its fit.
    
    Parameters:
        f: freq list
        P: PSDs
        frange: range in the frequency that is to be fitted
        plot: boolean to plot the PSD with its fit (default True)
    Output:
        gamma
    """
    p = np.polyfit(np.log(f[frange[0]: frange[1]]), np.log(P[frange[0]: frange[1]]), 1)
    if plot:
        plt.figure('Noise fit for range ' + str(f[frange[0]]) + ' - ' + str(f[frange[1]]) + 'Hz')
        plt.plot(f, P)
        plt.plot(f[frange[0]: frange[1]], np.exp(p[1])*f[frange[0]: frange[1]]**p[0])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (I**2/Hz)')
        
    return -p[0]


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


def spreadPlotter(currents, name = ''):
    """
    Plots the spread of the variance found of the sampled data versus the mean.
    The variance is calculated using np.var(), not a PSD integral.
    Fits y = ax + b through the datapoints
    Parameters:
        currents: measured currents
        name: name of the figure (optional)
    Returns:
        Ivar: variance of the currents
        a: slope of the fit
        b: offset of the fit
    """
    _, currents = removeClipping(currents)
    Imean = np.mean(currents, axis = 1)
    Ivar = np.var(currents, axis = 1)
    [a, b] = np.polyfit(Imean, Ivar**.5, 1)
    x = np.linspace(Imean[abs(Imean).argmin()], Imean[abs(Imean).argmax()], 100)    
    plt.figure(name)
    plt.plot(Imean, Ivar**.5, '.')
    plt.plot(x, a * x + b, '--') 
    plt.xlabel('$I_\mu$ (nA)')
    plt.ylabel('$I_\sigma$')
    plt.title('Standard deviations of CV configurations')
    plt.tight_layout()     
    return Ivar, a, b

def gaussFit(currents, n, m, bins = 100, CVname = ''):
    """
    Fits a Gaussian distribution to the given currents and plots them to give
    visual support.
    Parameters:
        currents: the output currents
        n: number of plots below each other
        m: number of plots next to each other
        bins: number of bins for histogram (default = 100)
        CVname: additional name to discriminate between datasets (optional)
    """
    plt.figure('Gaussian fit on data ' + CVname)
    for i in range(currents.shape[0]):
        plt.subplot(n,m,i+1)
        mu, sigma = scipy.stats.norm.fit(currents[i,:])
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, bins)
        plt.plot(x,mlab.normpdf(x, mu, sigma))
        plt.hist(currents[i,:], bins, normed = True)
        plt.title('$\mu$=' + str(round(mu, 3)) + ', $\sigma$=' + str(round(sigma, 3)))
    plt.tight_layout()
    plt.show()
    
def CVPlotter(controlVoltages, electrodes, variance):
    """
    Makes a 3D plot using two input electrodes the corresponding variance.
    
    Parameters:
        controlVoltages: 
        electrodes: The two inuts that are to be plotted (tuple)
        variance: variance of the current signal
    """
    fig = plt.figure('CV dependency plot on electrodes ' + str(electrodes[0]) + ' and ' + str(electrodes[1]))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(controlVoltages[:,electrodes[0]], controlVoltages[:,electrodes[1]], variance) #color = '#ff7f0e'
    ax.set_xlabel('CV ' + str(electrodes[0]))
    ax.set_ylabel('CV ' + str(electrodes[1]))
    ax.set_zlabel('$\sigma_I$')
    