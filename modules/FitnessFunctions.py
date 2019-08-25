# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:14:52 2019

@author: HCRuiz
"""
import numpy as np
from SkyNEt.modules.Classifiers import perceptron 

#TODO: implement corr_lin_fit (AF's last fitness function)?

#%% Accuracy of a perceptron as fitness: meanures separability 
def accuracy_fit(outputpool, target, clipvalue=np.inf):
    genomes = len(outputpool)
    fitpool = np.zeros(genomes)
    for j in range(genomes):
        output = outputpool[j]
    
        if np.any(np.abs(output)>clipvalue):
            acc = 0
            print(f'Clipped at {clipvalue} nA')
        else:
            x = output[:,np.newaxis]
            y = target[:,np.newaxis]
            acc, _, _ = perceptron(x,y)
            
        fitpool[j] = acc
    return fitpool

#%% Correlation between output and target: measures similarity
def corr_fit(outputpool, target, clipvalue=np.inf):
    genomes = len(outputpool)
    fitpool = np.zeros(genomes)
    for j in range(genomes):
        output = outputpool[j]
        if np.any(np.abs(output)>clipvalue):
            print(f'Clipped at {clipvalue} nA')
            corr = -1
        else:
            x = output[:,np.newaxis]
            y = target[:,np.newaxis]
            X = np.stack((x, y), axis=0)[:,:,0]
            corr = np.corrcoef(X)[0,1]
            
        fitpool[j] = corr
    return fitpool

#%% Combination of a sigmoid with pre-defined separation threshold (2.5 nA) and 
#the correlation function. The sigmoid can be adapted by changing the function 'sig( , x)'        
def corrsig_fit(outputpool, target, clipvalue=np.inf):
    
    genomes = len(outputpool)
    fitpool = np.zeros(genomes)
    for j in range(genomes):
        output = outputpool[j]
        if np.any(np.abs(output)>clipvalue):
            print(f'Clipped at {clipvalue} nA')
            fit = -100
        else:
            buff0 = target == 0
            buff1 = target == 1
            max_0 = np.max(output[buff0])
            min_1 = np.min(output[buff1])
            sep = min_1 - max_0
            x = output[:,np.newaxis]
            y = target[:,np.newaxis]
            X = np.stack((x, y), axis=0)[:,:,0]
            corr = np.corrcoef(X)[0,1]
            if sep >= 0: 
                fit = sig(sep) * corr
            else:
                fit = sig(sep) * corr * 0.01
        fitpool[j] = fit   
    return fitpool

#Sigmoid function. 
def sig(sep):
    return 1/(1+np.exp(-5*(sep/2.5-0.5)))+ 0.1
