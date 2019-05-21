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
from SkyNEt.modules.Nets.staNNet import staNNet
from SkyNEt.modules.Nets.lightNNet import lightNNet
from SkyNEt.modules.Nets.DataHandler import DataLoader as dl
from SkyNEt.modules.Nets.DataHandler import GetData as gtd
#%%
###############################################################################
########################### LOAD DATA  ########################################
###############################################################################

np.random.seed(22)
Seed = False
main_dir = r'C:\Users\User\APH\Thesis\Data\wave_search\paper_chip_dataset2\2019_05_17_095928_trainData_3d\data4nn\20_05_2019\\'
file_name = 'data_for_training_lightNNet.npz'
data = dl(main_dir, file_name, syst='cpu', steps=12)

generate_input = True
noisefit = False

#%%
###############################################################################
############################ DEFINE NN and RUN ################################
###############################################################################
depth = 5
width = 90

learning_rate,nr_epochs,batch_size = 3e-3, 10, [2048]

runs = 1
valerror = np.zeros((runs,nr_epochs))
trainerror = np.zeros((runs,nr_epochs))
beta1 = 0.9
beta2 = 0.75
for i in range(runs):
    if generate_input:
        net = lightNNet(data,depth,width)
    else:
        net = staNNet(data,depth,width)
    net.train_nn(learning_rate,nr_epochs,batch_size[i],betas=(beta1, beta2),seed=Seed)
    valerror[i] = net.L_val
    trainerror[i] = net.L_train
    print('Run nr. ',i)
    # Save every run so that they can be used to determine test error
    net.save_model(main_dir+'MSE_d'+ str(depth) + 'w90_'+str(nr_epochs)+'ep_lr3e-3_b'+str(batch_size[i])+'_b1b2_'+str(beta1) + str(beta2) + '.pt')
norm_valerror = valerror

#%%
###############################################################################
########################### LOAD NN & TEST ####################################
###############################################################################
generate_input = False

#file_dir = r'C:\Users\User\APH\Thesis\Data\wave_search\paper_chip_dataset2\testsets\2019_05_20_123116_test_set_7h\test_set_skip12.npz'
file_dir = r'C:\Users\User\APH\Thesis\Data\wave_search\paper_chip_dataset2\testsets\2019_05_20_202552_testset1_40k\testset1_40k.npz'

NN_dir = r'C:\Users\User\APH\Thesis\Data\wave_search\paper_chip_dataset2\2019_05_17_095928_trainData_3d\Nets\MSE\\'
NN_file = 'MSE_d5w90_500ep_lr3e-3_b[2048]_b1b2_0.90.75-21-05-04h05m.pt'

if generate_input:
    net = lightNNet(NN_dir + NN_file)
    net.info['phase'] = np.ones(7)
else:
    net = staNNet(NN_dir + NN_file)

########################## TEST GENERALIZATION  ###############################
inputs, targets = gtd(file_dir, syst='cpu') #function to load data returning torch Variable with correct form and dtype 
targets = targets
prediction = net.outputs(inputs)*net.info['conversion']


### Training profile
#plt.figure()
#plt.plot(np.arange(net.info['L_val'].shape[0]),net.info['L_val'].T)
#plt.title('Validation MSE Profile while Training')
#plt.xlabel('Epochs')
#plt.show()

### Test Error
subsample = np.random.permutation(len(prediction))[:100000]
plt.figure()
plt.subplot(1,2,1)
plt.plot(targets[subsample],prediction[subsample],'.')
plt.xlabel('True Output (nA)')
plt.ylabel('Predicted Output (nA)')
min_out = np.min(np.concatenate((targets[subsample],prediction[subsample])))
max_out = np.max(np.concatenate((targets[subsample],prediction[subsample])))
plt.plot(np.linspace(min_out,max_out),np.linspace(min_out,max_out),'k')
#plt.title('Predicted vs True values')

error = (targets[:]-prediction[:]).T#/np.sqrt(baseline_var)
print(f'MSE on Test Set: \n {np.mean(error**2)}')

plt.subplot(1,2,2)
plt.hist(error[subsample],100)
plt.xlabel('error (nA)')
plt.ylabel('nr. of samples')
#plt.title('Scaled error histogram')
plt.show()
