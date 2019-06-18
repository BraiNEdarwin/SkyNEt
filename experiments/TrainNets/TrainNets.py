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
main_dir = r'/home/student/Documents/Mark/NN_data/paper_device_27_04_2019/data4nn/12_06_2019_with_sines/'
#r'/home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/Devices/Marks_Data/April_2019/train set/data4nn/16_04_2019/'
file_name = 'data_for_training.npz'
data = dl(main_dir, file_name)
#mu_train = data[0][1].mean()
#sig_train = data[0][1].std()

#%%
###############################################################################
############################ DEFINE NN and RUN ################################
###############################################################################
depth = 5
width = 90
learning_rate,nr_epochs,batch_size =  1e-4,2000,128
#3e-4,1000,1024 for conversion=1 MSE=240 # try lr=3e-5
#beta1,beta2 = 0.9, 0.75
runs = 1
valerror = np.zeros((runs,nr_epochs))
trainerror = np.zeros((runs,nr_epochs))
for i in range(runs):
    net = staNNet(data,depth,width)
    net.train_nn(learning_rate,nr_epochs,batch_size)#,betas=(beta1,beta2))
    valerror[i] = net.L_val
    trainerror[i] = net.L_train
    print('Run nr. ',i)


### Training profile
plt.figure()
plt.plot(np.arange(nr_epochs),valerror.T, label='Val. Error')
plt.plot(np.arange(nr_epochs),trainerror.T, label='Train. Error')
plt.title('Validation MSE Profile while Training')
plt.xlabel('Epochs')
plt.legend()
plt.show()

now = datetime.datetime.now()
nowstr = now.strftime('%d-%m-%Hh%Mm')
save_dir = main_dir+f'{nowstr}_lr{learning_rate}-eps{nr_epochs}-mb{batch_size}/'#-b1{beta1}-b2{beta2}/'
#plt.savefig(save_dir+'Learning_profile.png')

#%%
###############################################################################
############################## SAVE NN ########################################
###############################################################################
path = save_dir+f'NN_{nowstr}.pt'
#net.save_model(path)
#Then later: net = staNNet(path)
# Save other stuff? e.g. generalization/test error...

#%%
###############################################################################
########################### LOAD NN & TEST ####################################
###############################################################################
#net = staNNet(path)


########################## TEST GENERALIZATION  ###############################
file_dir = r'/home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/Devices/Marks_Data/May_2019/'
inputs, targets = gtd(file_dir+'testset_uniform_random.npz', syst='cuda') #function to load data returning torch Variable with correct form and dtype 
prediction = net.outputs(inputs)
 

###################### ------- Basic Plotting ------- #######################

### Test Error
subsample = np.random.permutation(len(prediction))[:30000]
subsampl_t = targets[subsample]
subsampl_pred = prediction[subsample]
error = (targets-prediction)

plt.figure()
plt.subplot(1,2,1)
plt.plot(targets[subsample],prediction[subsample],'.')
plt.xlabel('True Output')
plt.ylabel('Predicted Output')
min_out = np.min(np.concatenate((subsampl_t,subsampl_pred)))
max_out = np.max(np.concatenate((subsampl_t,subsampl_pred)))
plt.plot(np.linspace(min_out,max_out),np.linspace(min_out,max_out),'k')
plt.title('Predicted vs True values')
print(f'MSE on Test Set: \n {np.mean(error**2)}')
plt.subplot(1,2,2)
plt.hist(np.reshape(error,error.size),100)
plt.title('Scaled error histogram')
plt.show()

plt.figure()
plt.plot(targets,error,'.')
plt.plot(np.linspace(targets.min(),targets.max(),len(error)),np.zeros_like(error))
plt.title('Error vs Output')
plt.xlabel('Output')
plt.ylabel('Error')

val_pred =  net.outputs(data[1][0])
val_targets = data[1][1].cpu().numpy().squeeze()
val_error = val_targets-val_pred

plt.figure()
plt.plot(val_targets,val_error,'.')
plt.plot(np.linspace(val_targets.min(),val_targets.max(),len(val_error)),np.zeros_like(val_error))
plt.title('Val. Error vs Output')
plt.xlabel('Output')
plt.ylabel('Error')
