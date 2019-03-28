#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:46:45 2018
This script is an example on how to construct and train a web of multiple neural networks to achieve a task.
@author: ljknoll
"""
import torch
import matplotlib.pyplot as plt
from SkyNEt.modules.Nets.predNNet import predNNet
from SkyNEt.modules.Nets.webNNet import webNNet

# create nn object from which the web is made
main_dir = r'C:\Users\User\APH\Thesis\Data\wave_search\setup2\2019_02_09_091323_device2_speedTest_1G_factor_0.1_T_86400s_batch_50s_fs_1000Hz\\'
data_dir = 'TEST_NN.pt'
net1 = predNNet(main_dir+data_dir)


# Initialize web object
web = webNNet()

# add networks as vertices
# network object, name of vertex
web.add_vertex(net1, 'A', output=True)
#web.add_vertex(net1, 'B')

# connect vertices with arcs, source->sink
# source vertex, sink vertex, sink gate index
#web.add_arc('B', 'A', 2)


# Check if web is valid (and optionally plot)
#web.check_graph(print_graph=True)

N = 400
batch_size = 2
max_epochs = 100

# explicitly set traindata for each network:
train_data = torch.zeros(N, 2)
train_data[200:,0] = 1
train_data[100:200,1] = 1
train_data[300:,1] = 1


# use same train data for all vertices:
#train_data = torch.zeros(N, 2)
#train_data[:,1] = 0.9

# target data 
targets = torch.ones(N, 1)
targets[100:300] = 0

# training
loss1, params1 = web.train(train_data, targets, batch_size, max_epochs, lr=0.05)

plt.figure()
plt.plot(loss1)
plt.xlabel('epochs')
plt.ylabel('MSE')
plt.title('training example with default settins')

# reset parameters of we
web.reset_parameters()

# OPTIONAL: define custom optimizer,
# see https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer
#optimizer = torch.optim.SGD
#loss2, params2 = web.train(train_data, targets, batch_size, nr_epochs, optimizer=optimizer, lr=0.01)
