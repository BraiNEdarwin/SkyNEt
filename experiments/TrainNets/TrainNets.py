#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 09:31:26 2018
This script is a template to train the NN on the data loaded by DataLoader(dir), which devides the 
data into training and validation sets and returns a tensor object used by the NN class. 
@author: hruiz
"""
import torch
import random
import numpy as np
from matplotlib import pyplot as plt
from SkyNEt.modules.Nets.staNNet import staNNet
from SkyNEt.modules.Nets.DataHandler import DataLoader as dl
from SkyNEt.modules.Nets.DataHandler import GetData as gtd
#%%
###############################################################################
########################### LOAD DATA  ########################################
###############################################################################
random.seed(22)
Seed = True
main_dir = r'C:\Users\User\APH\Thesis\Data\wave_search\2019_01_30_123621_characterization_20h_batch_25s_fs_500_f_2\\'
file_name = 'data_for_training_10percent_skip4.npz'
data, baseline_var = dl(main_dir, file_name, test_set=False)
factor = 2
#freq = torch.sqrt(torch.tensor([2,np.pi,5,7,13,17,19],dtype=torch.float32)) * factor
freq = np.sqrt(np.array([2,np.pi,5,7,13,17,19])) * factor
Vmax = 0.8
fs = 500
generate_input = True
phase = np.zeros(7)
#phase = torch.zeros(7,dtype=torch.float32)
#%%
###############################################################################
############################ DEFINE NN and RUN ################################
###############################################################################
depth = 5
width = 90
learning_rate,nr_epochs,batch_size = 3e-4, 10, 512
runs = 1
valerror = np.zeros((runs,nr_epochs))
for i in range(runs):
    if generate_input:
        net = staNNet(data,depth,width,freq,Vmax,fs,phase)
    else:
        net = staNNet(data,depth,width)
    net.train_nn(learning_rate,nr_epochs,batch_size,betas=(0.9, 0.75),seed=Seed)
    valerror[i] = net.L_val
    print('Run nr. ',i)

print('Baseline Var. is ', baseline_var)
norm_valerror = valerror/baseline_var

#%%
###############################################################################
############################## SAVE NN ########################################
###############################################################################
net.save_model(main_dir+'TEST_NN_25percent_skip7.pt')
#Then later: net = staNNet(path)
# Save other stuff? e.g. generalization/test error...

#%%
###############################################################################
########################### LOAD NN & TEST ####################################
###############################################################################
net = staNNet(main_dir+'TEST_NN_25percent.pt')

########################## TEST GENERALIZATION  ###############################
file_dir = main_dir+'test_set.npz'
factor = 2.3
freq = np.sqrt(np.array([2, np.pi, 5, 7, 13, 17, 19])) * factor
Vmax = 0.8
fs = 500
phase = np.zeros(7)
generate_input = True

inputs, targets = gtd(file_dir) #function to load data returning torch Variable with correct form and dtype 
if generate_input:
    prediction = net.outputs(inputs,freq,Vmax,fs,phase)
else:
    prediction = net.outputs(inputs)
 
#%%
###################### ------- Basic Plotting ------- #######################

### Training profile
plt.figure()
plt.plot(np.arange(nr_epochs),valerror.T)
plt.title('Validation MSE Profile while Training')
plt.xlabel('Epochs')
plt.show()

### Test Error
subsample = np.random.permutation(len(prediction))[:30000]
plt.figure()
plt.subplot(1,2,1)
plt.plot(targets[subsample],prediction[subsample],'.')
plt.xlabel('True Output')
plt.ylabel('Predicted Output')
min_out = np.min(np.concatenate((targets[subsample],prediction[subsample])))
max_out = np.max(np.concatenate((targets[subsample],prediction[subsample])))
plt.plot(np.linspace(min_out,max_out),np.linspace(min_out,max_out),'k')
plt.title('Predicted vs True values')

error = (targets[:]-prediction[:]).T#/np.sqrt(baseline_var)
plt.subplot(1,2,2)
plt.hist(error,100)
plt.title('Scaled error histogram')
plt.show()
