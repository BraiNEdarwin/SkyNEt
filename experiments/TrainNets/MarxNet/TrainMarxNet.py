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
main_dir = r'/home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/Marks_Data/April_2019/train set/data4nn/16_04_2019/'
file_name = 'data_for_training.npz'
data = dl(main_dir, file_name, steps=3)

#%%
###############################################################################
############################ DEFINE NN and RUN ################################
###############################################################################
lss = 'MSE'
depth = 20
width = 90
learning_rate,nr_epochs,batch_size = 3e-3, 500, 2048 #3e-3, 200, 1024 w. depth=10
# RMSE: 1.254 Error with d=20,w=90 and 3e-3, 500, 2048
# Doing run now for loss=MSE and same parameters as above: 1.23 Error, 
#so there is no difference between models yet... This can be do to the dominant term being the cross talk
beta1,beta2 = 0.9, 0.75
runs = 1
valerror = np.zeros((runs,nr_epochs))
trainerror = np.zeros((runs,nr_epochs))
for i in range(runs):
    net = staNNet(data,depth,width,loss=lss)
    net.train_nn(learning_rate,nr_epochs,batch_size,betas=(beta1,beta2))
    valerror[i] = net.L_val
    trainerror[i] = net.L_train
    print('Run nr. ',i)


### Training profile
plt.figure()
plt.plot(np.arange(nr_epochs),valerror.T, label='Val. Error')
plt.plot(np.arange(nr_epochs),trainerror.T, label='Train. Error')
plt.title('MSE Profile while Training')
plt.xlabel('Epochs')
plt.legend()

now = datetime.datetime.now()
nowstr = now.strftime('%d-%m-%Hh%Mm')
job_id = f'loss{lss}-d{depth}w{width}-lr{learning_rate}-eps{nr_epochs}-mb{batch_size}-b1{beta1}-b2{beta2}'
identifier = f'{nowstr}-Error_{job_id}'
plt.savefig(main_dir+identifier+'.png')
plt.show()
#%%
###############################################################################
############################## SAVE NN ########################################
###############################################################################
path = main_dir+f'{nowstr}_NN_{job_id}.pt'
net.save_model(path)
#Then later: net = staNNet(path)
# Save other stuff? e.g. generalization/test error...

#%%
###############################################################################
########################### LOAD NN & TEST ####################################
###############################################################################
net = staNNet(path)


########################## TEST GENERALIZATION  ###############################
file_dir = r'/home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/Marks_Data/April_2019/random_test_set/data4nn/16_04_2019/'
inputs, targets = gtd(file_dir+'data_for_test.npz') #function to load data returning torch Variable with correct form and dtype 
prediction = net.outputs(inputs)#*10 for RMSE
 

###################### ------- Basic Plotting ------- #######################
targets=np.mean(targets,axis=1,keepdims=True)
### Test Error
subsample = np.random.permutation(len(prediction))[:30000]
plt.figure()
plt.subplot(1,2,1)
plt.plot(targets[subsample],prediction[subsample],'.')
plt.xlabel('True Output')
plt.ylabel('Predicted Output')
min_out = np.min(np.concatenate((targets[subsample],prediction[subsample,np.newaxis]),axis=1))
max_out = np.max(np.concatenate((targets[subsample],prediction[subsample,np.newaxis]),axis=1))
plt.plot(np.linspace(min_out,max_out),np.linspace(min_out,max_out),'k')
plt.title('Predicted vs True values')

ERR = (targets[:,0]-prediction.T).T
MSE = np.mean(ERR**2)
error = np.sqrt(MSE)
print(f'Error on Test Set: \n {error}')
plt.subplot(1,2,2)
plt.hist(ERR,100)
plt.title(f'Error = {error}')

identifier = f'{nowstr}-TestError_{job_id}'
plt.savefig(main_dir+identifier+'.png')
plt.show()
