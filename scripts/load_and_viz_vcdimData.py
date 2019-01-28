#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 11:50:33 2019
Visualize the data obtained from the VCdim search
@author: hruiz
"""
import numpy as np
from matplotlib import pyplot as plt
import pprint as pp

dir_file = r'../results/VC_dim/Device/Capacity_N4/2019_01_22_18-14/'
#r'../results/VC_dim/Device/Capacity_N4/2019_01_21_17-04/'
#with np.load(dir_file+'Summary_Results.npz') as data:
#    for var,arr in data.items():
#        exec(var+'= arr')
#        print(var,'loaded; shape:',arr.shape)

data = np.load(dir_file+'Summary_Results.npz')

def get_badcases(data, threshold_fit = 1.0, threshold_acc = 0.8):
    binary_labels = data['binary_labels']
    fitness_classifier =  data['fitness_classifier']
    accuracy_classifier = data['accuracy_classifier']
    
    low_fitness = np.arange(len(binary_labels))[fitness_classifier<threshold_fit]   
    z=zip(binary_labels[low_fitness].tolist(),fitness_classifier[low_fitness].tolist())
    lowfit_list = [x for x in z]
    print('Low fitness cases:')
    pp.pprint(lowfit_list)
    
    low_accuracy = np.arange(len(binary_labels))[accuracy_classifier<threshold_acc]
    z=zip(binary_labels[low_accuracy].tolist(),accuracy_classifier[low_accuracy].tolist())
    lowacc_list = [x for x in z]
    print('Low accuracy cases:')
    pp.pprint(lowacc_list)
    
    lowfit_hiacc = [n for n in low_fitness if n not in low_accuracy]
    z=zip(binary_labels[lowfit_hiacc].tolist(),fitness_classifier[lowfit_hiacc].tolist())
    lfha_list = [x for x in z]
    print('Low fit, high acc:')
    pp.pprint(lfha_list)
    
    d = {'low acc': lowacc_list, 'low acc index': low_accuracy, 
         'low fit' : lowfit_list,'low fit index': low_fitness, 
         'lfha' : lfha_list, 'lfha index' : lowfit_hiacc}
    return d 

bad_dict = get_badcases(data)

gcl = data['genes_classifier']
accl = data['accuracy_classifier']
outcl = data['output_classifier']
fitcl = data['fitness_classifier']

plt.figure()
ax1 = plt.subplot2grid((4,2),(0,0),colspan=2)
ax2 = plt.subplot2grid((4,2),(1,0),rowspan=2)
ax3 = plt.subplot2grid((4,2),(1,1))
ax4 = plt.subplot2grid((4,2),(2,1))
for i,gene in enumerate(gcl[1:-1]):
    a = accl[i+1]
    c = tuple(gene)[:3]+(1,)
#    print(i,gene)
    ax1.plot(outcl[i+1].T, color=c)
    if i+1 in bad_dict['low acc index']:
        ax2.plot(gene,'o', color=c)
        ax3.plot(fitcl[i+1],accl[i+1],'o', color=c)
        ax4.plot(outcl[i+1],color=c)
    else:
        ax2.plot(gene,'d', color=c)
        ax3.plot(fitcl[i+1],accl[i+1],'d', color=c)
#ax5.plot(ex2_bestgenes)
plt.show()

from SkyNEt.modules.GenWaveform import GenWaveform
from SkyNEt.modules.Classifiers import perceptron
### ---> The cases with negative fitness are those were also the ones with high accuracy
### but incorrect labelling (this is due to the neg. weight[0] in the perceptron!!)
plt.figure()
w_arr = np.zeros_like(bad_dict['lfha index'],dtype=float)
for i,l in enumerate(bad_dict['lfha index']):
    y = outcl[l,:][:,np.newaxis]
    trgt = np.asarray(GenWaveform(data['binary_labels'][l], [125], slopes=[10]))[:,np.newaxis]
    accuracy, weights, predicted = perceptron(y,trgt)
    w_arr[i] = weights[0]
    plt.subplot(2,4,i+1)
    plt.plot(y,'r')
    plt.plot(trgt,'k')
    plt.plot(predicted[0],predicted[1],'xg')
    plt.text(265,0.5,'weights[0]=%.1f' % weights[0],ha='center',va='center')
plt.subplot(2,4,8)
plt.plot(w_arr,fitcl[bad_dict['lfha index']],'o')
plt.show()


data = np.load(dir_file+'2019_01_22_211159_VCdim-0110/data.npz')
output_arr = data['output']
fitness_arr = data['fitness']
plt.figure()
plt.subplot(2,1,1)
for i in range(7):
    n = np.random.randint(0,100)
    plt.plot(outout[n,::5,:].T)
plt.subplot(2,1,2)
plt.plot(fitness_arr)
plt.show()
data.close()

