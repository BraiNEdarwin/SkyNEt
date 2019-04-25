#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:08:57 2019

@author: hruiz
"""
import numpy as np
import torch
from torch.autograd import Variable
#from SkyNEt.modules.Nets.predNNet import predNNet
from SkyNEt.modules.Nets.predNNet import predNNet
from matplotlib import pyplot as plt
###############################################################################
########################### LOAD DATA  ########################################
###############################################################################
#Load all_wfms.txt (outputs), fitness.tx (genes) and input_wfms.txt
#main_dir = r'C:\Users\User\APH\Thesis\Data\wave_search\champ_chip\2019_04_05_172733_characterization_2days_f_0_05_fs_50\nets\MSE_n_d10\\'
#data_dir = main_dir+'' #'2018_08_07_164652_CP_FullSwipe/'

#net = predNNet( data_dir + 'MSE_n_d10w90_50ep_lr3e-3_b1024_b1b2_0.90.75_seed.pt' )
data_dir = r'C:\Users\User\APH\Thesis\Data\wave_search\champ_chip\2019_04_05_172733_characterization_2days_f_0_05_fs_50\nets\MSE_n_proper\\'
net = predNNet(data_dir + 'MSE_n_d10w90_300ep_lr3e-3_b1024_b1b2_0.90.75_seed.pt')
pred_logic = True
pred_ring = False

# Possible loss functions: mse, cor, cormse
training_type = 'cor'
input_voltages = np.array([2,4])  # Determines which indices
learning_rate,nr_epochs,batch_size = 9e-3, 400, 64
reg_scale = 40.0

N = 100
inp_upper = 0.0
inp_lower = -0.8
x_inp = inp_lower*np.ones((2,N*4))
x_inp[1, N:2*N] = inp_upper
x_inp[0, 2*N:3*N] = inp_upper
x_inp[0, 3*N:] = inp_upper
x_inp[1, 3*N:] = inp_upper

syst = 'cpu' # 'cpu' #
if syst is 'cuda':
    print('Train with CUDA')
    dtype = torch.cuda.FloatTensor
    itype = torch.cuda.LongTensor
else: 
    print('Train with CPU')
    dtype = torch.FloatTensor
    itype = torch.LongTensor

gates = ['AND','NAND','OR','NOR','XOR','XNOR']
# hardcoded target values of logic gates with off->lower and on->upper
upper = 1.
lower = 0.
###############################################################################
################# GENERATE INPUT AND TEST DATA FOR GATES ######################
###############################################################################
y_target = upper*np.ones((6, 4*N))
y_target[0, :3*N] = lower
y_target[1, 3*N:] = lower
y_target[2, :N] = lower
y_target[3, N:] = lower
y_target[4, :N] = lower
y_target[4, 3*N:] = lower
y_target[5, N:3*N] = lower

x_augm = x_inp.T
y_augm = y_target

###############################################################################
######################### RING DATA HANDLING ##################################
###############################################################################
if pred_ring:
    datafile = r'C:\Users\User\APH\Thesis\Data\wave_search\champ_chip\2019_03_14_143310_characterization_7D_t_4days_f_0_1_fs_100\ring_dataset\Ring_class_data_0.40_many.npz'       
    x_augm = np.load(datafile)['inp_wvfrm'] - 0.25
    #plt.plot(x_augm[:,0],x_augm[:,1],'.')
    y_augm = np.load(datafile)['target'][np.newaxis,:]*upper
    

###############################################################################
######################### GENERAL DATA HANDLING ###############################
###############################################################################
    
#noise = np.min(np.abs(np.diff(np.abs(np.unique(y_target,axis=1)))),axis=1)
#y_augm = y_augm + 0.1*noise[:,np.newaxis]*np.random.randn(y_augm.shape[0],y_augm.shape[1])
# Return everything to cpu
dtype = torch.FloatTensor
x = torch.from_numpy(x_augm).type(dtype)
y_target = torch.from_numpy(y_augm).type(dtype)
x = Variable(x)
y = Variable(y_target)

# Permute time point indices 
permutation = torch.randperm(y.data.shape[-1])#.type(itype)
Np_test = 52
test_indices = permutation[:Np_test]
train_indices = permutation[Np_test:]

x_train = x[train_indices]
y_train = y[:,train_indices,np.newaxis]
x_test = x[test_indices]
y_test = y[:,test_indices,np.newaxis]

pred_voltages = np.zeros((len(y),5))
y_pred = np.zeros(y.shape)
#error = np.zeros(len(y))
grads_epoch = np.zeros((len(y),nr_epochs,5))
valErr_pred = np.zeros((nr_epochs,len(y)))

###############################################################################
###################### SELF DEFINED LOSS FUNCTIONS ############################
###############################################################################
def cor_loss_fn(x, y):
    corr = torch.mean((x-torch.mean(x))*(y-torch.mean(y)))
    x_high_min = torch.min(x[(y == upper)]).item()
    x_low_max = torch.max(x[(y == lower)]).item()
    return (1.001 - (corr/(torch.std(x,unbiased=False)*torch.std(y,unbiased=False)+1e-10)))/(abs(x_high_min-x_low_max)/3)**0.5

mse_loss_fn = torch.nn.MSELoss()


if training_type == 'mse' or training_type == None:
    loss_fn = mse_loss_fn
    
elif training_type=='cor':
    def loss_fn(x, y):
        return cor_loss_fn(x[:,0], y[:,0])
    
elif training_type=='cormse':
    alpha = 0.8
    def loss_fn(x_in, y_in):
        x = x_in[:,0]
        y = y_in[:,0]
        cor = cor_loss_fn(x, y)
        mse = mse_loss_fn(x, y)
        #print (str(cor) + '   ' + str(mse))
        return alpha*cor+(1-alpha)*mse/(upper-lower)**2

###############################################################################
###################### PREDICT CONTROL VOLTAGES ###############################
############################################################################### 
net.loss_fn = loss_fn
net.inputs = input_voltages
net.model.cpu()
#net.loss_fn.cpu()


for i in range(len(y_pred)): 
#    select data to train the network with NN(I,net)=O
    if pred_logic: print('Predicting ',gates[i])
    else: print('Predicting ring...')
    data = [(x_train,y_train[i]),(x_test,y_test[i])]
    pred_voltages[i], valErr_pred[:,i] = net.predict(data, learning_rate,
                 nr_epochs, batch_size, reg_scale = reg_scale) #reg_scale=20 works! 
    #how to make a function that I call with data but uses also net?
    y_pred[i] = 10*net.predictor(x).data.numpy()[:,0] # network is trained on tens of nanoamps!

plt.figure()
for i in range(len(y_pred)): 
    plt.subplot(int(np.ceil(len(y_pred)/3)),int(np.ceil(len(y_pred)/2)),i+1)
    plt.plot(y_pred[i],'b')
    plt.plot(y_target[i].numpy(),':k')
    plt.xlabel('Time')
    plt.ylabel('Output')
    plt.legend(['Converged','Target'])
    plt.title(gates[i])


log10_error = np.log10(valErr_pred)
plt.figure()
plt.title('Learning Control Voltages')
plt.plot(np.arange(nr_epochs),log10_error) 
plt.ylabel('Normalized Validation Error log10')
plt.xlabel('Epochs')
plt.legend(gates)

#saveArrays(r'C:\Users\User\APH\Thesis\Data\wave_search\champ_chip\2019_04_05_172733_characterization_2days_f_0_05_fs_50\nets\MSE_n_adap_200ep\ring\\', filename="result_NAME,nr_epochs = nr_epochs,trained_cv=pred_voltages,input_electrodes = input_voltages,training_type=training_type,lr=learning_rate,input_scaling = '*0.7 -0.1',valErr_pred=valErr_pred,log10_error=log10_error,pred=y_pred,x_inp=x_augm)