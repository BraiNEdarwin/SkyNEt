#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:46:45 2018
This script is an example on how to construct and train a web of multiple neural networks to achieve a task.
@author: ljknoll
"""
import torch
import matplotlib.pyplot as plt
from SkyNEt.modules.Nets.staNNet import staNNet
from SkyNEt.modules.Nets.webNNet import webNNet

# create nn object from which the web is made
main_dir = r'/home/lennart/Dropbox/afstuderen/search_scripts/'
data_dir = 'lr2e-4_eps400_mb512_20180807CP.pt'
net1 = staNNet(main_dir+data_dir)


# Initialize web object
web = webNNet()

# add networks as vertices
# network object, name of vertex
web.add_vertex(net1, 'A', output=True)
web.add_vertex(net1, 'B')
web.add_vertex(net1, 'C')

# connect vertices with arcs, source->sink
# source vertex, sink vertex, sink gate index
web.add_arc('B', 'A', 2)
web.add_arc('C', 'A', 3)


# Check if web is valid (and optionally plot)
web.check_graph(print_graph=True)

N = 100
batch_size = 20
max_epochs = 20

# input data, size (N, 2 * nr of networks)
train_data = torch.cat((torch.zeros(N), torch.linspace(-0.9, 0.9, N))).view(2, -1).t()

# duplicate train_data for each network
train_data = torch.cat((train_data,)*len(web.graph), dim=1)

# target data, size (N, nr output vertices)
targets = torch.linspace(0, 0.5, N).view(-1,1)


# create plot function
def plot_results(loss, params, title, error_name):
    web.reset_parameters(params)
    web.forward(train_data)
    output = web.get_output()

    fig = plt.figure()
    fig.suptitle(title)
    
    plt.subplot(1, 2, 1)
    plt.plot(output)
    plt.plot(targets)
    plt.legend(['trained result', 'target'])

    plt.subplot(1, 2, 2)
    plt.plot(loss)
    plt.xlabel('epochs')
    plt.ylabel(error_name)
    
    plt.show()


# reset parameters of web
web.reset_parameters()

# training
loss1, params1 = web.train(train_data, targets, batch_size, max_epochs, lr=0.05)

plot_results(loss1, params1, 'training example with default settings (Adam and MSE)', 'MSE')


# OPTIONAL: define custom optimizer,
# see https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer
optimizer = torch.optim.SGD
web.reset_parameters()
loss2, params2 = web.train(train_data, targets, batch_size, max_epochs, optimizer=optimizer, lr=0.01)

plot_results(loss2, params2, 'training example custom optimizer SGD', 'MSE')


# OPTIONAL: define custom loss function
def loss_fn(y_pred, y):
    return torch.mean((y_pred-y)**4)

web.reset_parameters()
loss3, params3 = web.train(train_data, targets, batch_size, max_epochs, loss_fn=loss_fn, lr=0.05)

plot_results(loss3, params3, 'training example custom loss function', '4th power loss')