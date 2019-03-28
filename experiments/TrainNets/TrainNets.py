#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 09:31:26 2018
This script is a template to train the NN on the data loaded by DataLoader(dir), which devides the 
data into training and validation sets and returns a tensor object used by the NN class. 
@author: hruiz
"""
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
main_dir = r'C:\Users\User\APH\Thesis\Data\wave_search\champ_chip\2019_03_14_143310_characterization_7D_t_4days_f_0_1_fs_100\\'
file_name = 'data_for_training_skip3.npz'
data, baseline_var = dl(main_dir, file_name, test_set=False)
factor = 0.1
#freq = torch.sqrt(torch.tensor([2,np.pi,5,7,13,17,19],dtype=torch.float32)) * factor
freq = np.sqrt(np.array([2,np.pi,5,7,13,17,19])) * factor
amplitude = np.array([0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9])
offset = np.array([-0.3, -0.3, -0.2, -0.2, -0.2, -0.2, -0.2])
fs = 100
generate_input = True
noisefit = True
phase = np.zeros(7)
#phase = torch.zeros(7,dtype=torch.float32)
#%%
###############################################################################
############################ DEFINE NN and RUN ################################
###############################################################################
depth = 15
width = 90
learning_rate,nr_epochs,batch_size = 1e-2, 10, 1024
runs = 1
valerror = np.zeros((runs,nr_epochs))
for i in range(runs):
    if generate_input:
        net = staNNet(data,depth,width,freq,amplitude,fs,offset,phase,noisefit)
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
net.save_model(main_dir+'MSE_n_d15w90_10ep_lr1e-2_b1024.pt')
#Then later: net = staNNet(path)
# Save other stuff? e.g. generalization/test error...


#%%
###############################################################################
########################### LOAD NN & TEST ####################################
###############################################################################

net = staNNet(main_dir+'MSE_n_d15w90_10ep_lr1e-2_b1024.pt')

########################## TEST GENERALIZATION  ###############################
file_dir = r'C:\Users\User\APH\Thesis\Data\wave_search\champ_chip\7D_test_sets\2019_03_19_084109_rand_test_set_100ms\\'+'test_set.npz'
#file_dir = r'C:\Users\User\APH\Thesis\Data\wave_search\champ_chip\7D_test_sets\2019_03_18_173336_testset_7D_t_12h_f_0_087_fs_100\test_set_skip3.npz'
factor = 0.087
freq = np.sqrt(np.array([2, np.pi, 5, 7, 13, 17, 19])) * factor
amplitude = np.array([0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9])
offset = np.array([-0.3, -0.3, -0.2, -0.2, -0.2, -0.2, -0.2])
phase = np.ones(7)
fs = 100
#phase = np.zeros(7)
generate_input = False

inputs, targets = gtd(file_dir) #function to load data returning torch Variable with correct form and dtype 
targets = targets/10 #\\
if generate_input:
    prediction = net.outputs(inputs,freq,amplitude,fs,offset,phase)
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
