#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 12:37:08 2018

@author: ljknoll
"""

import torch
import numpy as np
from Nets.predNNet import predNNet
from Nets.webNNet import webNNet
import matplotlib.pyplot as plt

# data import
data_dir = r'/home/lennart/Desktop/2018_11_02_142235_CP_inputs_and_targets.npz'
data = np.load(data_dir)
input_data = torch.tensor(data['inputs']).float()
target_data = torch.tensor(data['target']).long() + 1

# network import
main_dir = r'/home/lennart/Desktop/nnweb/'
data_dir = 'lr2e-4_eps400_mb512_20180807CP.pt'
net1 = predNNet(main_dir+data_dir)

web = webNNet()
web.add_vertex(net1, 'A', output=True)
web.add_vertex(net1, 'B', output=True)
web.add_vertex(net1, 'C', output=True)

batch_size = 150
nr_epochs = 500
lr = 0.05
beta = 0.1
cv_reset = 'rand'
bias = True
scale = False

optimizer = torch.optim.Adam
betas=(0.9, 0.75)
eps=1e-08

cross_fn = torch.nn.CrossEntropyLoss()

def loss_fn(y_pred, y):
    return cross_fn(y_pred, y[:,0])

loss, best_cv = web.train(input_data, target_data, 
                     beta=beta,
                     batch_size=batch_size,
                     nr_epochs=nr_epochs,
                     optimizer=optimizer,
                     loss_fn=loss_fn,
                     lr = lr,
                     betas = betas,
                     eps = eps,
                     bias=bias,
                     scale=scale)

def plot_results(best_cv):
    web.reset_parameters(best_cv)
    s = web.scale.data
    b = web.bias.data
    output_data = web.output_data*(1+s)+b
    plt.figure()
    plt.plot(target_data)
#    plt.plot(torch.sigmoid(output_data))
    pred_targets = torch.zeros_like(target_data)
    for i,value in enumerate(output_data):
        pred_targets[i] = value.argmax()
    plt.plot(pred_targets)
#    plt.legend(['target', 'net1'+str(b[0]), 'net2'+str(b[1]), 'net3'+str(b[2]), 'net_classification'])
    plt.legend(['target', 'net_classification'])
    s_r = list(map(lambda x: round(x,1), s.tolist()))
    b_r = list(map(lambda x: round(x,1), b.tolist()))
    plt.title('CrossEntropyLoss with 3 parallel networks, scale:%s, bias:%s' % (s_r, b_r))
    plt.show()

plt.figure()
plt.plot(loss)
plt.show()

plot_results(best_cv)