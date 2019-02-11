#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 11:50:33 2019
Visualize the data obtained from the VCdim search
@author: hruiz
"""
from helper_vcdim import *
import numpy as np
from matplotlib import pyplot as plt
import pprint as pp
import os
import pdb 

dir_file = r'../results/VC_dim/Device/Capacity_N4/2019_01_21_17-04/' #2019_01_22_18-14/'
#r'../results/VC_dim/Device/Capacity_N4/2019_01_21_17-04/'
#with np.load(dir_file+'Summary_Results.npz') as data:
#    for var,arr in data.items():
#        exec(var+'= arr')
#        print(var,'loaded; shape:',arr.shape)

coi = ['1001','0110']

data = np.load(dir_file+'Summary_Results.npz')

lr_profile = get_profile(dir_file)
bad_dict = get_badcases(data)

gcl = data['genes_classifier']
accl = data['accuracy_classifier']
outcl = data['output_classifier']
fitcl = data['fitness_classifier']
binary_labels = data['binary_labels']


plt.figure()
ax = plt.subplot2grid((3,2),(0,0))
plt.title('Fitness of fittest')
ax1 = plt.subplot2grid((3,2),(0,1))
plt.title('Final Output')
ax2 = plt.subplot2grid((3,2),(1,0),rowspan=2)
plt.ylabel('Gene value')
plt.xticks([0,1,2,3,4],['#1','#2','#3','#4','#5'],rotation=90)
plt.xlabel('Control Voltages')
ax3 = plt.subplot2grid((3,2),(1,1))
plt.xlabel('Fitness')
plt.ylabel('Accuracy')
ax4 = plt.subplot2grid((3,2),(2,1))
plt.title('Low accuracy examples')
      
for i,gene in enumerate(gcl[1:-1]):
#    a = accl[i+1]
    c = tuple(gene)[:3]+(1,)
    lb = str(binary_labels[i+1]).lstrip('[').rstrip(']')
    lb = ''.join(lb.split())
    indx = [x==lb for x in lr_profile['labels']]
    ax.plot(lr_profile['max fitness'][:,indx],color=c)
    ax1.plot(outcl[i+1].T, color=c)
    if i+1 in bad_dict['low acc index']:
        ax2.plot(gene,'o', color=c)
#        labels = np.asarray(bad_dict['low acc'])[i+1==bad_dict['low acc index']][0][0]
        ax3.plot(fitcl[i+1],accl[i+1],'o', color=c,label=str(lb))
        ax4.plot(outcl[i+1],color=c)
#        plot_top_genes(lr_profile,i,labels)
    else:
        ax2.plot(gene,'d', color=c)
        ax3.plot(fitcl[i+1],accl[i+1],'d', color=c) 
    if lb in coi: plot_top_genes(lr_profile,lb)
    
ax3.legend(fontsize='x-small')
plt.tight_layout()
plt.show()

data.close()

#from SkyNEt.modules.GenWaveform import GenWaveform
#from SkyNEt.modules.Classifiers import perceptron
#### ---> The cases with negative fitness are those were also the ones with high accuracy
#### but incorrect labelling (this is due to the neg. weight[0] in the perceptron!!)
#plt.figure()
#w_arr = np.zeros_like(bad_dict['lfha index'],dtype=float)
#for i,l in enumerate(bad_dict['lfha index']):
#    y = outcl[l,:][:,np.newaxis]
#    trgt = np.asarray(GenWaveform(data['binary_labels'][l], [125], slopes=[10]))[:,np.newaxis]
#    accuracy, weights, predicted = perceptron(y,trgt)
#    w_arr[i] = weights[0]
#    plt.subplot(2,4,i+1)
#    plt.plot(y,'r')
#    plt.plot(trgt,'k')
#    plt.plot(predicted[0],predicted[1],'xg')
#    plt.text(265,0.5,'weights[0]=%.1f' % weights[0],ha='center',va='center')
#plt.subplot(2,4,8)
#plt.plot(w_arr,fitcl[bad_dict['lfha index']],'o')
#plt.xlabel('weight[0] value')
#plt.ylabel('Fitness')
#plt.show()