#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:29:42 2019

@author: ljknoll

Script to plot the output of the device while sweeping one gate (m points) keeping the others constant
This is done for 50000/m different configurations of the remaining gate voltages

"""

import torch
import numpy as np
from SkyNEt.modules.Nets.staNNet import staNNet
from SkyNEt.modules.Nets.webNNet import webNNet
import matplotlib.pyplot as plt


# number of random control voltages to check
m = 500
n = 500000//m

# prepare web
main_dir = r'/home/lennart/Dropbox/afstuderen/search_scripts/'
network_dir = 'NN_skip3_MSE.pt'
net1 = staNNet(main_dir+network_dir)

output = torch.zeros(7, n, m)
for g in range(0,7):
    input_gates = [g]

    # generate control voltage gates
    control_gates = []
    for i in range(7):
        if i not in input_gates:
            control_gates.append(i)

    web = webNNet()
    web.add_vertex(net1, 'A', output=True, input_gates=control_gates)
    
    tmin = net1.info['offset'][input_gates]-net1.info['amplitude'][input_gates]
    tmax = net1.info['offset'][input_gates]+net1.info['amplitude'][input_gates]
    input_voltages = torch.linspace(np.asscalar(tmin),np.asscalar(tmax), m)
    
    # create random tensor of all voltages to be applied
    scaling = torch.FloatTensor(net1.info['amplitude'][control_gates]*2)
    offset = torch.FloatTensor(net1.info['offset'][control_gates]) - scaling/2
    cv = torch.rand(n,len(control_gates))*scaling + offset
    
    # feed voltages through device for each of the points
    with torch.no_grad():
        for i,point in enumerate(input_voltages):
            web.reset_parameters({'A':point})
            output[g,:,i] = web.forward(cv)[:,0]
    

def plot_output(output):
    for g,out in enumerate(output):
        plt.figure()
        plt.plot(input_voltages.repeat(n).numpy(), out.flatten().numpy(), ',k', alpha=0.3)
        plt.ylabel('output (nA)')
        plt.xlabel('voltage on gate %i (V)' % g)
        plt.title('output space characterization on gate %i, random sampling' % g)

plot_output(output)