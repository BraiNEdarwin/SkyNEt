#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 12:18:29 2018
Wrapper to measure the VC dimension of a device using the measurement script measure_VCdim.py
This wrapper creates the binary labels for N points and for each label it finds the control voltages.
If successful (measured by a threshold on the correlation and by the perceptron accuracy), the entry 1 is set in a vector corresponding to all labellings.
@author: hruiz
"""
from create_binary import bintarget
import evolve_VCdim as vcd
import numpy as np
from matplotlib import pyplot as plt
import time
import os

# Create binary labels for N samples
inputs = [[-1,1,-1,1],[-1,-1,1,1]]
N=len(inputs[0])
binary_labels = bintarget(N)  
threshold = 1-(0.65/N)*(1+1.0/N)
#Initialize container variables
fitness_classifier = []
genes_classifier = []
output_classifier = []
accuracy_classifier = []
found_classifier = []

#Create save directory
filepath0 = r'../../test/evolution_test/'
date = time.strftime('%Y_%m_%d-%H%M')
filepath1 = filepath0+'VCdim_testing/'+date
dirname = filepath1+'_Capacity/'
if os.path.exists(filepath0):
    os.makedirs(dirname)
else:
    assert 1==0, 'No directory created. Parent target directory '+filepath0+' does not exist'
    
for bl in binary_labels:
    
    if len(set(bl))==1:
        print('Label ',bl,' ignored')
        genes, output, fitness, accuracy = np.nan, np.nan, np.nan, np.nan
        found_classifier.append(1)
    else:
        genes, output, fitness, accuracy = vcd.evolve(inputs,bl, filepath=filepath1)
        if accuracy>threshold:
            found_classifier.append(1)
        else:
            found_classifier.append(0)
        
    genes_classifier.append(genes)
    output_classifier.append(output)
    fitness_classifier.append(fitness)
    accuracy_classifier.append(accuracy)
    
fitness_classifier = np.array(fitness_classifier)
accuracy_classifier = np.array(accuracy_classifier)
found_classifier = np.array(found_classifier)
capacity = np.mean(found_classifier)

for i in range(len(genes_classifier)):
    if genes_classifier[i] is np.nan:
        genes_classifier[i] = np.nan*np.ones_like(genes_classifier[1])
        output_classifier[i] = np.nan*np.ones_like(output_classifier[1])
        
output_classifier = np.array(output_classifier)
genes_classifier = np.array(genes_classifier)

plt.figure()
plt.plot(fitness_classifier,accuracy_classifier,'o')
plt.xlabel('Fitness')
plt.ylabel('Accuracy')
plt.show()

np.savez(dirname+'VCdim_'+str(N)+'-samples', 
         binary_labels = binary_labels,
         capacity = capacity,
         found_classifier = found_classifier,
         fitness_classifier = fitness_classifier,
         accuracy_classifier = accuracy_classifier,
         output_classifier = output_classifier,
         genes_classifier = genes_classifier)