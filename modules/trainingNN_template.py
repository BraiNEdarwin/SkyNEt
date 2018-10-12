#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 09:31:26 2018
This script is a template to train the NN on the data loaded by DataLoader(dir), which devides the 
data into training and validation sets and returns a tensor object used by the NN class. 
@author: hruiz
"""

import numpy as np
from matplotlib import pyplot as plt
from Nets.predNNet import predNNet
from Nets.DataHandler import DataLoader as dl
from Nets.DataHandler import GetData as gtd
#%%
###############################################################################
########################### LOAD DATA  ########################################
###############################################################################
main_dir = r'/home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/data4nn/2018_10_12/'
file_name = 'data_for_training.npz'
data, baseline_var = dl(main_dir, file_name, test_set=True)

#%%
###############################################################################
############################ DEFINE NN and RUN ################################
###############################################################################
depth = 5
width = 90
learning_rate,nr_epochs,batch_size = 3e-4, 100, 512
runs = 1
valerror = np.zeros((runs,nr_epochs))
for i in range(runs):
    net = predNNet(data,depth,width)
    net.train_nn(learning_rate,nr_epochs,batch_size,betas=(0.9, 0.75))
    valerror[i] = net.L_val
    print('Run nr. ',i)

print('Baseline Var. is ', baseline_var)
norm_valerror = valerror/baseline_var

#%%
###############################################################################
############################## SAVE NN ########################################
###############################################################################
net.save_model(main_dir+'TEST_NN.pt')
#Then later: net = predNNet(path)
# Save other stuff? e.g. generalization/test error...

#%%
###############################################################################
########################### LOAD NN & TEST ####################################
###############################################################################
net = predNNet(main_dir+'TEST_NN.pt')

########################## TEST GENERALIZATION  ###############################
file_dir = main_dir+'test_set_from_trainbatch.npz'
inputs, targets = gtd(file_dir) #function to load data returning torch Variable with correct form and dtype 
prediction = net.outputs(inputs)

#TODO: Two scripts where the CV prediction using GD and GA are done respectively
 
#%%
###################### ------- Basic Plotting ------- #######################

### Training profile
plt.figure()
plt.plot(np.arange(nr_epochs),norm_valerror.T)
plt.title('Norm. Validation MSE Profile while Training')
plt.xlabel('Epochs')
plt.show()

### Test Error
subsample = np.random.permutation(len(prediction))[:30000]
plt.figure()
plt.subplot(1,2,1)
plt.plot(targets[subsample],prediction[subsample],'.')
plt.xlabel('True Output')
plt.ylabel('Predicted Output')
min_out = np.min(np.concatenate((targets[subsample],prediction[subsample,np.newaxis])))
max_out = np.max(np.concatenate((targets[subsample],prediction[subsample,np.newaxis])))
plt.plot(np.linspace(min_out,max_out),np.linspace(min_out,max_out),'k')
plt.title('Predicted vs True values')

error = (targets[:,0]-prediction.T).T/np.sqrt(baseline_var)
plt.subplot(1,2,2)
plt.hist(error,100)
plt.title('Scaled error histogram')
plt.show()
