#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 11:50:33 2019
Visualize the data obtained from the VCdim search
@author: hruiz
"""
import numpy as np
from matplotlib import pyplot as plt

dir_file = r'../results/VC_dim/Device/Capacity_N4/2019_01_22_18-14/'
#r'../results/VC_dim/Device/Capacity_N4/2019_01_21_17-04/'
with np.load(dir_file+'Summary_Results.npz') as data:
    for var,arr in data.items():
        exec(var+'= arr')
        print(var,'loaded; shape:',arr.shape)

threshold_fit = 1.0
threshold_acc = 0.8

low_fitness = np.arange(len(binary_labels))[fitness_classifier<threshold_fit]
low_accuracy = np.arange(len(binary_labels))[accuracy_classifier<threshold_acc]

z=zip(binary_labels[low_accuracy].tolist(),accuracy_classifier[low_accuracy].tolist())
lowacc_list = [x for x in z]
print('Low accuracy cases: \n',lowacc_list)

lowfit_hiacc = [n for n in low_fitness if n not in low_accuracy]
z=zip(binary_labels[lowfit_hiacc].tolist(),fitness_classifier[lowfit_hiacc].tolist())
lfha_labels = [x for x in z]
print('Low fit, high acc: \n',lfha_labels)

plt.figure()
ax1 = plt.subplot2grid((4,2),(0,0),colspan=2)
ax2 = plt.subplot2grid((4,2),(1,0))
ax3 = plt.subplot2grid((4,2),(1,1))
for i,gene in enumerate(genes_classifier[1:-1]):
    a = accuracy_classifier[i+1]
    c = tuple(gene)[:3]+(a**3,)
    print(i,gene)
    ax1.plot(output_classifier[i+1].T, color=c)
    ax2.plot(gene,':o', color=c)
    ax3.plot(fitness_classifier[i+1],accuracy_classifier[i+1],'o', color=c)
#    ax4.plot(max_fitness[i+1],color=c)
#ax4.plot(ex1_bestgenes)
#ax5.plot(ex2_bestgenes)
plt.show()

from SkyNEt.modules.GenWaveform import GenWaveform
from SkyNEt.modules.Classifiers import perceptron
### ---> The cases with negative fitness are those were also the ones with high accuracy
### but incorrect labelling (this is due to the neg. weight[0] in the perceptron!!)
plt.figure()
w_arr = np.zeros_like(lowfit_hiacc,dtype=float)
for i,l in enumerate(lowfit_hiacc):
    y = output_classifier[l,:][:,np.newaxis]
    trgt = np.asarray(GenWaveform(binary_labels[l], [125], slopes=[10]))[:,np.newaxis]
    accuracy, weights, predicted = perceptron(y,trgt)
    w_arr[i] = weights[0]
    plt.subplot(2,4,i+1)
    plt.plot(y,'r')
    plt.plot(trgt,'k')
    plt.plot(predicted[0],predicted[1],'xg')
    plt.text(265,0.5,'weights[0]=%.1f' % weights[0],ha='center',va='center')
plt.subplot(2,4,8)
plt.plot(w_arr,fitness_classifier[lowfit_hiacc],'o')
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

