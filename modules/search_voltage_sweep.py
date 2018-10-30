#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 13:22:08 2018

@author: ljknoll

21 configurations of input:
I0 : {-0.9, 0, 0.9}
I1 : {-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9}
"""

import torch
from Nets.predNNet import predNNet
from Nets.webNNet import webNNet
import matplotlib.pyplot as plt
import matplotlib

main_dir = r'/home/lennart/Desktop/nnweb/'
data_dir = 'lr2e-4_eps400_mb512_20180807CP.pt'
net1 = predNNet(main_dir+data_dir)

web = webNNet()
web.add_vertex(net1, 'A', output=True)


N = 100 # determines step size of sweep: 1/N, must be int because sizes are then N+1
default_param = 0.8

I0 = [-0.9, 0, 0.9]
I1 = [-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9]

input_data = []
for i in I0:
    for j in I1:
        input_data.append([i,j])
input_data = torch.tensor(input_data)

# select which index to sweep over
for cv_index in range(5):
    # sweep over control voltage cv_index
    output_data = torch.empty(input_data.shape[0], N+1)
    for v in range(N+1):
        voltage = v/N
        cv = default_param*torch.ones(5)
        cv[cv_index] = voltage
        web.reset_parameters(cv)
        out = web.forward(input_data)
        output_data[:,v] = out[:,0].data
    
    # plotting
    cmap = matplotlib.cm.get_cmap('seismic')
    for (i, data) in enumerate(output_data):
        plt.plot([j/N for j in range(N+1)], data.tolist(), c=cmap(i%7/7))
        if i%7==0:
            plt.ylabel('network output')
            plt.xlabel('control voltage %s' % cv_index)
            plt.title('constant I0 = %s' % round(input_data[i,0].item(), 2))
            plt.show()