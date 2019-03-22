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

from SkyNEt.modules.Nets.predNNet import predNNet
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
main_dir = r'/home/lennart/Dropbox/afstuderen/search_scripts/'
data_dir = 'lr2e-4_eps400_mb512_20180807CP.pt'
net = predNNet(main_dir+data_dir)

web = webNNet()
web.add_vertex(net, 'A', output=True)

N = 100 # number of data points of one input, total 4*N

training_type = 'cormse'

# hardcoded target values of logic gates with off->lower and on->upper
target_hardcoded = True
upper = 1.0
lower = 0.0
# if set to false, use output of known cv configurations as targets


add_noise = False
sigma = 0.01 # standard deviation of noise added to target

# ------------------------ END configure ------------------------




# input data for both I0 and I1
input_data = torch.zeros(N*4,2)
input_data[N:2*N,   1] = 0.9
input_data[2*N:3*N, 0] = 0.9
input_data[3*N:,    0] = 0.9
input_data[3*N:,    1] = 0.9

gates = ['AND','NAND','OR','NOR','XOR','XNOR']

list_cv = torch.FloatTensor(
      [[387,-387,650,55,-892],[477,-234,-332,-358,827],
       [9,183,714,-313,-416],[514,665,-64,855,846],
       [-771,342,900,-655,-48],[480,149,-900,-2,-450]])
list_cv += 900
list_cv /=1800
cv_data = torch.ones(6,4*N)
for (i, cv) in enumerate(list_cv):
    # set parameters of network to cv
    web.reset_parameters(cv)
    # evaluate network
    cv_data[i] = web.forward(input_data).data[:,0]

if target_hardcoded:
    target_data = upper*torch.ones(6, 4*N)
    target_data[0, :3*N] = lower
    target_data[1, 3*N:] = lower
    target_data[2, :N] = lower
    target_data[3, N:] = lower
    target_data[4, :N] = lower
    target_data[4, 3*N:] = lower
    target_data[5, N:3*N] = lower
else:
    target_data = torch.tensor(cv_data)

def cor_loss_fn(x, y):
    corr = torch.mean((x-torch.mean(x))*(y-torch.mean(y)))
    return 1-corr/torch.std(x)/torch.std(y)
mse_loss_fn = torch.nn.MSELoss()

if training_type == 'bin':
    target_data = target_data.long()
    def loss_fn(y_p, y):
        y_pred = y_p - 0.3
        y_pred = y_pred*10
        y_pred = torch.cat((-y_pred, y_pred), dim=1)
        return cross_fn(y_pred, y[:,0])
    add_noise = False
elif training_type=='mse':
    loss_fn = mse_loss_fn
elif training_type=='cor':
    loss_fn = cor_loss_fn
elif training_type=='cormse':
    def loss_fn(x, y):
        alpha = 0.6
        mse = mse_loss_fn(x, y)
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
    for j in range(5):
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
        legend_list = ['target', 'network', 'cv_output']
        plt.plot(target_data[i].numpy())
        web.reset_parameters(store_cvs[i])
        store_output[i] = web.forward(input_data).data[:,0]
        plt.plot(store_output[i])
        plt.plot(cv_data[i].data)
        plt.legend(legend_list)
        plt.title("%s, cv:%s" % (gate, np.round(store_cvs[i], 3)))
    # adjust margins 
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # fullscreen plot   
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
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
