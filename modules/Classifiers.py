#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:42:27 2018

@author: hruiz
"""

import numpy as np
from matplotlib import pyplot as plt
#import pdb
def perceptron(wvfrm,target,tolerance=0.01,max_iter=200):
    #Assumes that the waveform wvfrm and the target have the shape (n_total,1)
    # Normalize the data; it is assumed that the target has binary values
    wvfrm = (wvfrm-np.mean(wvfrm))/np.std(wvfrm)
    n_total = len(wvfrm)
    weights = np.random.randn(2,1) #np.zeros((2,1))
    inp = np.concatenate([np.ones_like(wvfrm),wvfrm],axis=1)
    shuffle = np.random.permutation(len(inp))
    
    n_test = int(0.1*n_total)
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
    
    predict = np.array(list(map(f,np.dot(inp,weights))))
    buffer = np.zeros_like(target)
#    print(predict.shape,target.shape,inp.shape)
    buffer[target[:,0]==predict] = 1
    n_correct = np.sum(buffer)
    accuracy = n_correct/n_total
    
    predicted = predict
#    print('Fraction of iterations used: ', j/max_iter)
#    pdb.set_trace()
    corrcoef = np.corrcoef(target.T,inp[:,1].T)[0,1]
    if corrcoef<-0.001:
        print('Correlation is negative: ', corrcoef)
        accuracy = 0.
        print('Accuracy is set to zero!')
        
        
    return accuracy, weights, predicted

if __name__=='__main__':
    
    #XOR as target
    target = np.zeros((800,1))
    target[:200] = 1 
    
    #Create wave form
    noise = 0.05
    output = np.zeros((800,1))
    output[200:] = 1 
#    output[600:] = 1.75 
    wvfrm = output + noise*np.random.randn(len(target),1)
    
    accuracy, weights, predicted = perceptron(wvfrm,target)
    
    plt.figure()
    plt.plot(target)
    plt.plot(wvfrm,'.')
    plt.plot(np.arange(len(target)),(-weights[0]/weights[1])*np.ones_like(target),'g')
    plt.plot(predicted,'xk')
    plt.show()
    
    nr_examples = 100
    accuracy = np.zeros((nr_examples,))
    for l in range(nr_examples):
        accuracy[l], weights, _ =  perceptron(wvfrm,target)
        print(f'Prediction Accuracy: {accuracy[l]} and weights:{weights}')
        
    plt.figure()
    plt.hist(accuracy,100)
    plt.show()