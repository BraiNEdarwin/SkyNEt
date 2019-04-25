#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ljknoll

Searching for boolean logic functionality using Genetic Algorithm in web framework
Boolean logic gates
I0 I1    AND NAND OR NOR XOR XNOR
0  0     0   1    0  1   0   1
0  1     0   1    1  0   1   0
1  0     0   1    1  0   1   0
1  1     1   0    1  0   0   1
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from SkyNEt.modules.Nets.staNNet import staNNet
from SkyNEt.modules.Nets.webNNet import webNNet
import SkyNEt.experiments.boolean_logic.config_evolve_NN as config


# ------------------------ START configure ------------------------
cf = config.experiment_config()
cf.generations = 50
cf.mutationrate = 0.3
cf.fitnessavg = 1

# Save settings - not used at the moment
cf.filepath = r'../../test/evolution_test/NN_testing/'
cf.name = 'lineartest'

# Initialize NN
main_dir = r'C:\Users\User\APH\Thesis\Data\wave_search\champ_chip\2019_04_05_172733_characterization_2days_f_0_05_fs_50\nets\MSE_n_adap_200ep\\'
data_dir = 'MSE_n_adap_d10w90_200ep_lr3e-3_b1024_b1b2_0.90.75_seed.pt'
net = staNNet(main_dir+data_dir)

web = webNNet()
web.add_vertex(net, 'A', output=True)

N = 100 # number of data points of one input, total 4*N

training_type = 'cormse'

# hardcoded target values of logic gates with off->lower and on->upper
target_hardcoded = True
upper = 5.0
lower = 0.0
# if set to false, use output of known cv configurations as targets


add_noise = False
sigma = 0.01 # standard deviation of noise added to target

# ------------------------ END configure ------------------------




# input data for both I0 and I1
input_data = -0.5*torch.zeros(N*4,2)
input_data[N:2*N,   1] = 0.0
input_data[2*N:3*N, 0] = 0.0
input_data[3*N:,    0] = 0.0
input_data[3*N:,    1] = 0.0

gates = ['AND','NAND','OR','NOR','XOR','XNOR']

if training_type == 'bin':
    upper = 1
    lower = 0
target_data = upper*torch.ones(6, 4*N)
target_data[0, :3*N] = lower
target_data[1, 3*N:] = lower
target_data[2, :N] = lower
target_data[3, N:] = lower
target_data[4, :N] = lower
target_data[4, 3*N:] = lower
target_data[5, N:3*N] = lower

def cor_loss_fn(x, y):
    corr = torch.mean((x-torch.mean(x))*(y-torch.mean(y)))
    return 1.0-corr/(torch.std(x,unbiased=False)*torch.std(y,unbiased=False)+1e-16)
mse_loss_fn = torch.nn.MSELoss()
def mse_norm_loss_fn(y_pred, y):
    return mse_loss_fn(y_pred, y)/(upper-lower)**2

if training_type == 'bin':
    target_data = target_data.long()
    def loss_fn(y_p, y):
        y_pred = y_p - 0.3
        y_pred = y_pred*10
        y_pred = torch.cat((-y_pred, y_pred), dim=1)
        return cross_fn(y_pred, y[:,0])
    add_noise = False
elif training_type=='mse':
    loss_fn = mse_norm_loss_fn
elif training_type=='cor':
    loss_fn = cor_loss_fn
elif training_type=='cormse':
    def loss_fn(x, y):
        alpha = 0.1
        mse = mse_norm_loss_fn(x, y)
        cor = cor_loss_fn(x, y)
        return alpha*cor+(1-alpha)*mse
else:
    assert False, 'Specify loss function'

if add_noise:
    gauss = torch.distributions.Normal(0.0, sigma)
    target_data += gauss.sample((6, 4*N))


store_fitness = np.zeros((len(gates), cf.generations))
store_output = np.zeros((len(gates), input_data.shape[0]))
store_cvs = np.zeros((len(gates), 5))
for i,gate in enumerate(gates):
    print(i, gate)
    if training_type == 'bin':
        weights = torch.bincount(target_data[i,:]).float()
        weights = 1/weights
        weights /= torch.sum(weights)
        cross_fn = torch.nn.CrossEntropyLoss(weight = weights)
    
    best_error = 1e10
    for j in range(1):
        web.reset_parameters()
        geneArray, outputArray, fitnessArray = web.trainGA(input_data, 
                                                           target_data[i].view(-1,1), 
                                                           cf,
                                                           loss_fn = loss_fn)    
        print('best error of session %i: %.3f' % (j+1, min(-fitnessArray[:,0])))
        if min(-fitnessArray[:,0])<best_error:
            best_error = min(-fitnessArray[:,0])
            # control voltages
            store_cvs[i] = geneArray[-1,0]
            # error
            store_fitness[i] = -fitnessArray[:,0]
            # best output
            store_output[i] = outputArray[-1, 0, :, 0]

# printing
def print_gates():
    plt.figure()
    for i, gate in enumerate(gates):
        print(gate)
        # print output network and targets
        plt.subplot(2, 3 , 1 + i//2 + i%2*3)
        legend_list = ['target', 'network']
        plt.plot(target_data[i].numpy())
        web.reset_parameters(store_cvs[i])
        store_output[i] = web.forward(input_data).data[:,0]
        plt.plot(store_output[i])
        plt.legend(legend_list)
        plt.title("%s, cv:%s" % (gate, np.round(store_cvs[i], 3)))
    # adjust margins 
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()

print_gates()


def print_errors():
    # print training error
    plt.figure()
    for i,gate in enumerate(gates):
        plt.subplot(2, 3 , 1 + i//2 + i%2*3)
        plt.plot(store_fitness[i])
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title(gate)
    plt.show()

print_errors()
