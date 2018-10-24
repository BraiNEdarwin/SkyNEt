#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 21:44:10 2018
@author: ljknoll

Boolean logic gates
I0 I1    AND OR NAND NOR XOR XNOR
0  0     0   0  1    1   0   1
0  1     0   1  1    0   1   0
1  0     0   1  1    0   1   0
1  1     1   1  0    0   0   1
"""

import torch
from Nets.predNNet import predNNet
from Nets.webNNet import webNNet
import matplotlib.pyplot as plt
main_dir = r'/home/lennart/Desktop/nnweb/'
data_dir = 'lr2e-4_eps400_mb512_20180807CP.pt'
net1 = predNNet(main_dir+data_dir)

web = webNNet()
web.add_vertex(net1, 'A', output=True)
#web.add_vertex(net1, 'B')
#web.add_arc('B', 'A', 2)


N = 10

# input data for both I0 and I1
input_data = torch.zeros(N*4,2)
input_data[N:2*N,   1] = 0.9
input_data[2*N:3*N, 0] = 0.9
input_data[3*N:,    0] = 0.9
input_data[3*N:,    1] = 0.9


# target data for all gates
gates = ['AND','NAND','OR','NOR','XOR','XNOR']
if False:
    # hardcoded target values of logic gates with off->lower and on->upper
    upper = 0.9
    lower = 0.0
    target_data = upper*torch.ones(4*N,6)
    target_data[:3*N, 0] = lower
    target_data[3*N:, 1] = lower
    target_data[:N, 2] = lower
    target_data[N:, 3] = lower
    target_data[:N, 4] = lower
    target_data[3*N:, 4] = lower
    target_data[N:3*N, 5] = lower
else:
    # use output of known cv configurations as targets
    list_cv = torch.FloatTensor(
          [[387,-387,650,55,-892],[477,-234,-332,-358,827],
           [9,183,714,-313,-416],[514,665,-64,855,846],
           [-771,342,900,-655,-48],[480,149,-900,-2,-450]])
    list_cv += 900
    list_cv /=1800
    
    target_data = torch.ones(4*N,6)
    
    for (i, cv) in enumerate(list_cv):
        # set parameters of network to cv
        web.reset_parameters(cv)
        # evaluate network
        target_data[:, i] = web.forward(input_data).data[:,0]

# for each logic gate, train and plot difference
trained_cv = []
for (i,gate) in enumerate(gates):
    print(i, gate)
    loss = web.train(input_data, target_data[:,i], beta=0.01, maxiterations=200, lr = 0.0001, momentum=0.7, nesterov=True)
    trained_cv.append([i.data.tolist() for i in web.parameters()][0])
    
    # print training error
    plt.figure()
    plt.plot(loss)
    plt.xlabel('epochs')
    plt.ylabel('MSE loss')
    plt.title(gate)
    
    print(loss[-1])
    
    # print output of network and targets
    plt.figure()
    plt.plot(web.forward(input_data).data)
    plt.plot(target_data[:,i])
    plt.legend(['network', 'target'])
    plt.title(gate)