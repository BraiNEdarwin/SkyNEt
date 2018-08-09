#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:42:27 2018

@author: hruiz
"""

import numpy as np
from matplotlib import pyplot as plt

def perceptron(wvfrm,target,tolerance=0.01,max_iter=200,debug=False):
    #Assumes that the waveform wvfrm and the target have the shape (n_total,1)
    n_total = len(wvfrm)
    weights = 5*np.random.randn(2,1) #np.zeros((2,1))
    inp = np.concatenate([np.ones_like(wvfrm),wvfrm],axis=1)
    shuffle = np.random.permutation(len(inp))
    
    n_test = int(0.5*n_total)
    x_test = inp[shuffle[:n_test]]
    y_test = target[shuffle[:n_test]]
    
    x = inp[shuffle[n_test:]]
    y = target[shuffle[n_test:]]
    
    f = lambda x: float(x<0)
    error = np.inf
    j=0
    while (error>tolerance) and (j < max_iter):
        
        for i in range(len(x)):
            a = np.dot(weights.T,x[i])
            delta = y[i] - f(a)
            weights = weights - delta*x[i][:,np.newaxis]
            
        predict = np.array(list(map(f,np.dot(x_test,weights))))
        predict = predict[:,np.newaxis]
        error = np.mean(np.abs(y_test - predict))
        j += 1
#        print('Prediction Error: ',error, ' in ', j,' iters')
    
    buffer = np.zeros_like(y_test)
    buffer[y_test==predict] = 1
    n_correct = np.sum(buffer)
    accuracy = n_correct/n_test
    
    if debug:
        plt.figure()
        plt.plot(target)
        plt.plot(wvfrm,'.')
        plt.plot(np.arange(len(target)),-weights[0]*np.ones_like(target)/weights[1])
        plt.plot(shuffle[:n_test],predict,'xk')
        plt.show()

    return accuracy