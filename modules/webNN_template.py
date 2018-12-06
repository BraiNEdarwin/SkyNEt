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
main_dir = r'/home/lennart/Dropbox/afstuderen/search_scripts/'
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

N = 10
batch_size = 2
nr_epochs = 100

# explicitly set traindata for each network:
train_data = torch.zeros(N, 4)
train_data[:,1] = 0.2
train_data[:,3] = 0.3

# use same train data for all vertices:
#train_data = torch.zeros(N, 2)
#train_data[:,1] = 0.9

# target data 
targets = 0.5*torch.ones(N, 1)

# training
loss1, params1 = web.train(train_data, targets, batch_size, nr_epochs, lr=0.05)

# reset parameters of we
web.reset_parameters()

# OPTIONAL: define custom optimizer,
# see https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer
optimizer = torch.optim.SGD
loss2, params2 = web.train(train_data, targets, batch_size, nr_epochs, optimizer=optimizer, lr=0.01)

web.reset_parameters()

# OPTIONAL: define custom loss function
targets = torch.ones(N,).long()
torch_loss_fn = torch.nn.CrossEntropyLoss()
def loss_fn(y_pred, y):
    y_pred = torch.cat((y_pred, -y_pred), dim=1)
    return torch_loss_fn(y_pred, y)
loss3, params3 = web.train(train_data, targets, batch_size, nr_epochs, loss_fn=loss_fn, lr=0.05)