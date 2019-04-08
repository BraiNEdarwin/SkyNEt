#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 09:31:26 2018
This script is a template to train the NN on the data loaded by DataLoader(dir), which devides the 
data into training and validation sets and returns a tensor object used by the NN class. 
@author: hruiz
"""

import numpy as np
import datetime
from matplotlib import pyplot as plt
from SkyNEt.modules.Nets.staNNet import staNNet
from SkyNEt.modules.Nets.DataHandler import DataLoader as dl
from SkyNEt.modules.Nets.DataHandler import GetData as gtd
#%%
###############################################################################
########################### LOAD DATA  ########################################
###############################################################################
main_dir = r'D:\UTWENTE\PROJECTS\DARWIN\Data\Mark\\'
file_name = 'data_with_inputs.npz'
data = dl(main_dir, file_name, test_size = 0.1)

#%%
###############################################################################
############################ DEFINE NN and RUN ################################
###############################################################################
depth = 5
width = 90
learning_rate,nr_epochs,batch_size = 3e-4, 100, 512
beta1, beta2 = 0.9, 0.75
runs = 1
valerror = np.zeros((runs,nr_epochs))
for i in range(runs):
    net = staNNet(data,depth,width)
    net.train_nn(learning_rate,nr_epochs,batch_size,betas=(beta1,beta2))
    valerror[i] = net.L_val
    print('Run nr. ',i)

### Training profile
plt.figure()
plt.plot(np.arange(nr_epochs),valerror.T)
plt.title('Norm. Validation MSE Profile while Training')
plt.xlabel('Epochs')
plt.show()

now = datetime.datetime.now()
nowstr = now.strftime('%Y-%m-%d-%H%M')

plt.savefig(main_dir+f'{nowstr}-Error_lr{learning_rate}-eps{nr_epochs}-mb{batch_size}-b1{beta1}-b2{beta2}.png')
#%%
###############################################################################
############################## SAVE NN ########################################
###############################################################################
path = main_dir+f'TEST_NN_{nowstr}.pt'
net.save_model(main_dir+f'TEST_NN_{nowstr}.pt')
#Then later: net = staNNet(path)
# Save other stuff? e.g. generalization/test error...

#%%
###############################################################################
########################### LOAD NN & TEST ####################################
###############################################################################
net = staNNet(path)

########################## TEST GENERALIZATION  ###############################
file_dir = main_dir+'test_set_from_trainbatch.npz'
inputs, targets = gtd(file_dir) #function to load data returning torch Variable with correct form and dtype 
prediction = net.outputs(inputs)

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

error = (targets[:,0]-prediction.T).T
print(f'MSE on Test Set: \n {np.mean(error**2)}')
plt.subplot(1,2,2)
plt.hist(error,100)
plt.title('Scaled error histogram')
plt.show()
