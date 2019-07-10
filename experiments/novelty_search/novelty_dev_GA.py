#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 17:06:41 2019

@author: ljknoll

demonstration of novelty based searching in simulated devices
for purpose of visualisation, 1D output is created for 2 inputs, effectively creating 2d output.

"""
import torch
import matplotlib.pyplot as plt

from SkyNEt.modules.Nets.predNNet import predNNet
from SkyNEt.modules.Nets.webNNet import webNNet
import SkyNEt.experiments.boolean_logic.config_evolve_NN as config


# ------------------------ configure ------------------------
cf = config.experiment_config()
cf.generations = 500
cf.mutationrate = 0.3
cf.fitnessavg = 1

# Save settings - not used at the moment
cf.filepath = r'../../test/evolution_test/NN_testing/'
cf.name = 'novelty_test'

# Initialize NN
main_dir = r'/home/lennart/Dropbox/afstuderen/search_scripts/'
data_dir = 'lr2e-4_eps400_mb512_20180807CP.pt'
net = predNNet(main_dir+data_dir)

web = webNNet()
web.add_vertex(net, 'A', output=True)

initial_archive_size = 50
novelty_threshold = 0.2


# ------------------------ END configure ------------------------

index = initial_archive_size


def get_input(N):
    # input data for both I0 and I1
    input_data = -0.9*torch.ones(2*N,2)
    input_data[N:, 0] = 0.9
    input_data[:N, 1] = 0.9
    return input_data

input_data = get_input(1)

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


def plot_output(archive_output):        
    plt.figure()
    # archive
    plt.scatter(archive_output[:,0,0].numpy(), archive_output[:,1,0].numpy(), alpha=0.3)
    
    archive_random = torch.zeros_like(archive_output)
    for i in range(len(archive_output)):
        web.reset_parameters('rand')
        web.forward(input_data)
        archive_random[i] = web.get_output()
    # random archive
    plt.scatter(archive_random[:,0,0].numpy(), archive_random[:,1,0].numpy(), alpha=0.4)
    
    plt.xlabel('output with I0: %.1f V, I1: %.1f V' % (input_data[0,0].item(), input_data[0,1].item()))
    plt.ylabel('output with I0: %.1f V, I1: %.1f V' % (input_data[-1,0].item(), input_data[-1,1].item()))
    plt.legend(['novel', 'random'])

plot_output(archive_output)

# loop through archive and recalculate outputs
#archive_real_output = torch.zeros_like(archive_output)
#with torch.no_grad():
#    for i, cf in enumerate(archive):
#        web.set_parameters_from_pool(cf)
#        web.forward(input_data)
#        archive_real_output[i] = web.get_output()
#plot_output(archive_real_output, 'output')

