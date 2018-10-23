#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 23:09:58 2018

@author: lennart
"""

import torch
from Nets.predNNet import predNNet
from Nets.webNNet import webNNet

main_dir = r'/home/lennart/Desktop/nnweb/'
data_dir = 'lr2e-4_eps400_mb512_20180807CP.pt'
net1 = predNNet(main_dir+data_dir)
web = webNNet()
web.add_vertex(net1, 'A', output=True)
web.add_vertex(net1, 'B')
web.add_arc('B', 'A', 2)


N = 10

def generate_linear(batch_size, slope, offset):
    if slope+offset>1:
        print("WARNING: input values outside range of control voltages [0,1]!")
    x = torch.arange(0, 1, 1/N)
    y = slope*x+offset
    return torch.cat((x.view(-1,1), y.view(-1,1)), dim=1)

train_data[:,1] = 0.9

# target data 
targets = 0.5*torch.ones(N,1)

loss = web.train(train_data, targets, beta=0.01)