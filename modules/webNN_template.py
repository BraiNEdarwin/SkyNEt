#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:46:45 2018
This script is an example on how to construct and train a web of multiple neural networks to achieve a task.
@author: ljknoll
"""
import torch
from Nets.predNNet import predNNet
from Nets.webNNet import webNNet

# create nn object from which the web is made
main_dir = r'/home/lennart/Desktop/nnweb/'
data_dir = 'lr2e-4_eps400_mb512_20180807CP.pt'
net1 = predNNet(main_dir+data_dir)


# Initialize web object
web = webNNet()

# add networks as vertices
# network object, name of vertex
web.add_vertex(net1, 'A', output=True)
web.add_vertex(net1, 'B')

# connect vertices with arcs, source->sink
# source vertex, sink vertex, sink gate index
web.add_arc('B', 'A', 2)

# Check if web is valid (and optionally plot)
#web.check_graph(print_graph=True)

N = 10  # batch_size

# explicit train data for each network:
#train_data = torch.zeros(N, 4)
#train_data[:,1] = 0.2
#train_data[:,3] = 0.3

# use same train data for all vertices:
train_data = torch.zeros(N,2)
train_data[:,1] = 0.9

# target data 
targets = 0.5*torch.ones(N,1)

# training
loss = web.train(train_data, targets, lr=0.01)

# reset parameters of web
web.reset_parameters()

# OPTIONAL: define custom optimizer,
# see https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer
optimizer = torch.optim.Adam
loss = web.train(train_data, targets, optimizer=optimizer, lr=0.01)
plt.plot(loss)

web.reset_parameters()

# OPTIONAL: define custom loss function
#loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
loss_fn = torch.nn.L1Loss(reduction='sum')
loss = web.train(train_data, targets, optimizer=optimizer, loss_fn=loss_fn, lr=0.01)