# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:14:52 2019

@author: HCRuiz
"""
import numpy as np
from SkyNEt.modules.Classifiers import perceptron 

#TODO: include max and AF fitness functions

def accuracy_fit(outputpool, target, clip=np.inf):
    genomes = len(outputpool)
    fitpool = np.zeros(genomes)
    for j in range(genomes):
        output = outputpool[j]
    
        if np.any(np.abs(output)>clip):
            acc = 0
            print(f'Clipped at {clip} nA')
        else:
            x = output[:,np.newaxis]
            y = target[:,np.newaxis]
            acc, _, _ = perceptron(x,y)
            
        fitpool[j] = acc
    return fitpool

def corr_fit(outputpool, target, clip=np.inf):
    genomes = len(outputpool)
    fitpool = np.zeros(genomes)
    for j in range(genomes):
        output = outputpool[j]
        if np.any(np.abs(output)>clip):
            print(f'Clipped at {clip} nA')
            corr = -1
        else:
            x = output[:,np.newaxis]
            y = target[:,np.newaxis]
            X = np.stack((x, y), axis=0)[:,:,0]
            corr = np.corrcoef(X)[0,1]
            
        fitpool[j] = corr
    return fitpool
