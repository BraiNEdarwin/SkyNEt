#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 18:11:13 2019
Wrapper to estimate the VC dimension of a nn using PyTorch.
This wrapper creates the binary labels for N points and for each label it finds the weights.
If successful (measured by a threshold on the correlation and by the perceptron accuracy),
the entry 1 is set in a vector corresponding to all labellings.
@author: hruiz
"""
from create_binary import bintarget
import time
import dnVCdim_trainer as vcd
#from SkyNEt.modules.Nets.net_collection import single_layer_net as NN
from SkyNEt.modules.Nets.dopantNet import dopantNet as NN
import torch
import numpy as np
from matplotlib import pyplot as plt
import os

inputs = [[-0.6,0.6,-0.6,0.6,-0.3],[-0.6,-0.6,0.6,0.6,0.]]
# [[-0.7,0.7,-0.7,0.7,-0.35,0.35,0.,0.],[-0.7,-0.7,0.7,0.7,0.,0.,-1.0,1.0]] for device

N=len(inputs[0])
#Create save directory
filepath0 = r'../../results/VC_dim'

filepath1 = filepath0+'/Model/Capacity_N'+str(N)
date = time.strftime('%Y_%m_%d_%H-%M_Run3-2019_09_11_12-58')
dirname = filepath1+'/'+date+'/'
if os.path.exists(filepath0):
    os.makedirs(dirname)
else:
    assert 1==0, 'No directory created. Parent target directory '+filepath0+' does not exist'
    
## Create binary labels for N samples
bad_gates = 'yes' # None # 
if bad_gates is None:
    binary_labels = bintarget(N).tolist()
else:
    bad_gates_dir = filepath1+f'/2019_09_11_13-18_Run2.1-2019_09_11_12-58/'
    bad_gates = np.load(bad_gates_dir+'Summary_Results.npz')['indx_nf']
    binary_labels = bintarget(N)[bad_gates].tolist() 
    print(f'Missed classifiers ({len(bad_gates)}) in previous run: \n {bad_gates}')

threshold = (1-0.5/N)
print('Threshold for acceptance is set at: ',threshold)

#NN parameters
 # hidden_neurons for single_layer_net
 # list with input indices
nn_params = [1,2]
#Define loss function 
    # torch.nn.BCELoss() for single_layer_net
    # vcd.neg_sig_corr for dopantNet
loss_fn  = vcd.neg_sig_corr

#Initialize container variables
cost_classifier = []
weights_classifier = []
output_classifier = []
accuracy_classifier = []
found_classifier = []
    
for bl in binary_labels:
    if len(set(bl))==1:
        print('Label ',bl,' ignored')
        weights, output, cost, accuracy = np.nan, np.nan, np.nan, np.nan
        found_classifier.append(1)
        cost_classifier.append(cost)
    else:
        print('Finding classifier ',bl)
        #Initialize net
        net = NN(nn_params)
        #Train net
        weights, output, cost, accuracy, _ = vcd.train(inputs,bl,net, loss_fn)
        del net
        print("Acuracy: ", accuracy, " cost: ", cost[-1])
        print("Weights : \n",weights)
        if accuracy>threshold:
            found_classifier.append(1)
            print('FOUND!')
        else:
            found_classifier.append(0)
            print('NOT FOUND!')
        cost_classifier.append(cost[-1])
        
    weights_classifier.append(weights)
    output_classifier.append(output)
    accuracy_classifier.append(accuracy)
    
cost_classifier = np.array(cost_classifier)
accuracy_classifier = np.array(accuracy_classifier)
found_classifier = np.array(found_classifier)
capacity = np.mean(found_classifier)

for i in range(len(weights_classifier)):
    if weights_classifier[i] is np.nan:
        weights_classifier[i] = np.nan*np.ones_like(weights_classifier[1])
        output_classifier[i] = np.nan*np.ones_like(output_classifier[1])
        
output_classifier = np.array(output_classifier)
weights_classifier = np.array(weights_classifier)

not_found = found_classifier==0
if not_found.size > 0:
    try:
        indx_nf = np.arange(2**N)[not_found]
    except IndexError as error:
        print(f'{error} \n Trying indexing bad_gates')
        indx_nf = bad_gates[not_found]
    print('Classifiers not found: %s' % indx_nf)
    binaries_nf = np.array(binary_labels)[not_found]
    print('belongs to : \n', binaries_nf)
else:
    print('All classifiers found!')
    indx_nf = None

np.savez(dirname+'Summary_Results',
         inputs = inputs,
         binary_labels = binary_labels,
         capacity = capacity,
         found_classifier = found_classifier,
         cost_classifier = cost_classifier,
         accuracy_classifier = accuracy_classifier,
         output_classifier = output_classifier,
         weights_classifier = weights_classifier,
         threshold = threshold,
         indx_nf = indx_nf)

plt.figure()
plt.plot(cost_classifier,accuracy_classifier,'o')
plt.plot(np.linspace(np.nanmin(cost_classifier),np.nanmax(cost_classifier)),threshold*np.ones_like(np.linspace(0,1)),'-k')
plt.xlabel('cost')
plt.ylabel('Accuracy')
plt.show()

if bad_gates is None:
    plt.figure()
    plt.hist(cost_classifier[1:-1])
    plt.show()
else:
    plt.figure()
    plt.hist(cost_classifier)
    plt.show()

mask = (cost_classifier[1:-1]>0.5)*(accuracy_classifier[1:-1]>threshold)
plt.figure()
plt.plot(output_classifier[1:-1][mask,:,0].T,'-o')
plt.legend(np.asarray(binary_labels)[1:-1][mask])
plt.show()

try:
    output_nf = output_classifier[not_found,:,0]
    #plt output of failed classifiers
    plt.figure()
    plt.plot(output_nf.T,'-o')
    plt.legend(binaries_nf)
    
except:
    print('Error in plotting output!')