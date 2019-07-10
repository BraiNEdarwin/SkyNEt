#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:18:18 2019

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


# ------------------------ configure ------------------------
cf = config.experiment_config()
cf.generations = 1000
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

N = 1 # number of data points of one input, total 4*N
initial_archive_size = 10
novelty_threshold = 1.0

lower = 0.0
upper = 1.0

# ------------------------ END configure ------------------------

index = initial_archive_size

def get_bool_target(N, lower = 0.0, upper = 1.0):
    target_data = upper*torch.ones(6, 4*N)
    target_data[0, :3*N] = lower
    target_data[1, 3*N:] = lower
    target_data[2, :N] = lower
    target_data[3, N:] = lower
    target_data[4, :N] = lower
    target_data[4, 3*N:] = lower
    target_data[5, N:3*N] = lower
    return target_data

def get_bool_input(N):
    # input data for both I0 and I1
    input_data = torch.zeros(N*4,2)
    input_data[N:2*N,   1] = 0.9
    input_data[2*N:3*N, 0] = 0.9
    input_data[3*N:,    0] = 0.9
    input_data[3*N:,    1] = 0.9
    return input_data

input_data = get_bool_input(N)

geneArray, outputArray, fitnessArray, archive, archive_output, novelty_threshold = web.noveltyGA(  input_data, 
                                                                                                   cf, 
                                                                                                   initial_archive_size, 
                                                                                                   novelty_threshold, 
                                                                                                   normalize=False,
                                                                                                   verbose=True)

best_error = min(-fitnessArray[:,0])
# control voltages
store_cvs = geneArray[-1,0]
# error
store_fitness = -fitnessArray[:,0]
# best output
store_output = outputArray[-1, 0, :, 0]


def plot_all():
    # plot all outputs in archive, real outputs and normalized outputs
    plt.figure()
    plt.plot(archive_output.numpy().T)
    plt.title('normalized outputs')
    
    archive_real_output = torch.zeros_like(archive_output)
    with torch.no_grad():
        for i, cf in enumerate(archive):
            web.set_parameters_from_pool(cf)
            web.forward(input_data)
            archive_real_output[i] = web.get_output()[:,0]
    
    plt.figure()
    plt.plot(archive_real_output.numpy().T)
    plt.title('real outputs')
#plot_all()



# search for good boolean logic gate behaviour in archive
gates = ['AND','NAND','OR','NOR','XOR','XNOR']
target_data = get_bool_target(N)

def cor_fitness(x, y):
    # calculates correlations of x with all y along dimension 0 (size m)
    # x (n), y(m*n)
    corr = torch.mean(x*y, dim=1)-torch.mean(x)*torch.mean(y, dim=1)
    return corr/torch.std(x)/torch.std(y, dim=1)

def separation_fitness(x):
    # returns maximum separation of x along dimension 0 (size m)
    # x (m*n)
    maxs, _ = torch.max(x, 1)
    mins, _ = torch.min(x, 1)
    return torch.sigmoid((maxs-mins-0.3)*10)
    


def get_bool_cv_output(archive_output, alpha):
    boolean_output = torch.zeros(6,archive_output.shape[1])
    boolean_cv = torch.zeros(6,5)
    separation = separation_fitness(archive_output[:,:,0])
    for i,gate in enumerate(gates):
        fitness = alpha*cor_fitness(target_data[i], archive_output[:,:,0]) + separation*(1-alpha)
        best_index = torch.nonzero(fitness == torch.max(fitness))[0].item()
        boolean_output[i] = archive_output[best_index,:,0]
        boolean_cv[i] = torch.tensor(archive[best_index], dtype=torch.float32)
    return boolean_cv, boolean_output    

def print_gates(plot_output = True, plot_normalized=False):
    n = 100
    long_data = get_bool_input(n)
    long_target = get_bool_target(n)
    
    plt.figure()
    for i, gate in enumerate(gates):    
        web.reset_parameters(boolean_cv[i])
        output_data = web.forward(long_data).data         
        
        # print output network and targets
        plt.subplot(2, 3 , 1 + i//2 + i%2*3)
        plt.plot(long_target[i])
        legend_list = ['target']
        
        if plot_output:
            plt.plot(output_data)
            legend_list.append('network')
        
        if plot_normalized:
            plt.plot(0.1+(output_data-torch.min(output_data))/1.2/(max(output_data)-min(output_data)))
            legend_list.append('normalized network')
        
        plt.legend(legend_list)
        plt.title("%s, diff:%.3f, cv:%s" % (gate, max(output_data)-min(output_data), np.round(boolean_cv[i].numpy(), 3)))
    # adjust margins
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # fullscreen plot (only available with matplotlib auto)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()


boolean_cv, boolean_output = get_bool_cv_output(archive_output, alpha = 1.0)
print_gates(True, True)
