#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 14:20:58 2019
Functio0ns to generate data in 2-dim feature space
@author: hruiz
"""
import numpy as np
from matplotlib import pyplot as plt

def ring(N, R_out=0.5, R_in=.1, epsilon=0.1):
    '''Generates labelled data of a ring with class 1 and the center with class 0
    '''
    samples = -1 + 2*np.random.rand(N,2)
    norm = np.sqrt(np.sum(samples**2,axis=1))
    labels = np.empty(samples.shape[0])
#    labels[norm>R_out+epsilon/2] = 0
    labels[norm<R_in-epsilon/2] = 0
    labels[(norm<R_out-epsilon/2)*(norm>R_in+epsilon/2)] = 1
    #Deal with outliers
    labels[norm>R_out] = np.nan
    sample_0 = samples[labels==0]
    sample_1 = samples[labels==1]
    return sample_0,sample_1

def luna(N, a1=1, b1=0, a2=1.5, b2=0.3):
    samples = -1 + 2*np.random.rand(N,2)
    f1 = lambda x: a1*np.sqrt(x)
    f2 = lambda x: a2*np.sqrt(x)
    buff = samples[samples[:,0]>b1]
    buff = buff[np.abs(buff[:,1])<f1(buff[:,0])]
    
    mask = (buff[:,0] > b2)*(np.abs(buff[:,1])<f2(buff[:,0]-b2))
    buff = buff[~mask]
    return buff

def cross(N, width=0.25, length=0.8):
    samples = (-1 + 2*np.random.rand(N,2))*length/2
    buff1 = samples[np.abs(samples[:,0])<width/2]
    buff2 = samples[(np.abs(samples[:,1])<width/2)*(np.abs(samples[:,0])>width/2)]
    buff = buff1.tolist()+buff2.tolist()
    return np.array(buff)

if __name__ is '__main__':
    N = 1000 #Nr of samples
    ###############################
    #Sample from a ring
    ###############################
    sample_0, sample_1 = ring(N)
    #Subsample the largest class###
    nr_samples = min(len(sample_0),len(sample_1))
    max_array = max(len(sample_0),len(sample_1))
    indices = np.random.permutation(max_array)[:nr_samples]
    if len(sample_0) == max_array:
        sample_0 = sample_0[indices]
    else: 
        sample_1 = sample_1[indices]
        
    plt.figure()
    plt.plot(sample_0[:,0],sample_0[:,1],'.b')
    plt.plot(sample_1[:,0],sample_1[:,1],'.r')
    plt.show()
    
    ##############################
    #### Sample from a moon #####
    #############################
    l1 = luna(N)
    l2 = -luna(N)
    l2[:,0] += 0.9
    l2[:,1] += 0.75
    plt.figure()
    plt.plot(l1[:,0],l1[:,1],'.b')
    plt.plot(l2[:,0],l2[:,1],'.r')
    
    #############################
    #### Sample from a cross ####
    #############################
    cx = cross(N)
    plt.figure()
    plt.plot(cx[:,0],cx[:,1],'.')
    plt.show()