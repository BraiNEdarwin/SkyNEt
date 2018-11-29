#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 12:37:08 2018

@author: ljknoll
3 parallel networks using crossentropy
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

batch_size = 450
nr_epochs = 200
lr = 0.09
beta = 0.1
cv_reset = 'rand'
bias = True
scale = True

optimizer = torch.optim.Adam
betas=(0.9, 0.99)
eps=1e-08

cross_fn = torch.nn.CrossEntropyLoss()

def loss_fn(y_pred, y):
    return cross_fn(y_pred, y[:,0])

#start_cv = {'bias': torch.tensor([-5.3005,  4.1657, -1.4890]),
# 'scale': torch.tensor([23.3486,  6.6760, 20.4856]),
# 'A': torch.tensor([0.3116, 1.1165, 1.0337, 0.3012, 0.8443]),
# 'B': torch.tensor([0.3980, 0.8098, 0.6797, 0.9330, 0.0268]),
# 'C': torch.tensor([ 0.6142,  0.6194,  0.9121,  0.3109, -0.0953])}
#web.reset_parameters(start_cv)

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

def plot_results(best_cv, ind = False):
    web.reset_parameters(best_cv)
    web.forward(input_data)
    output_data = web.get_output().data
    if ind:
        plt.figure()
        plt.plot(output_data)
        plt.legend(['net1', 'net2', 'net3'])
        plt.show()
    plt.figure()
    plt.plot(target_data.float()-0.01)
    pred_targets = torch.zeros_like(target_data).float()
    for i,value in enumerate(output_data):
        pred_targets[i] = value.argmax().float()+0.01
    plt.plot(pred_targets)
    plt.legend(['target', 'net_classification'])
    s = web.scale.data
    b = web.bias.data
    s_r = list(map(lambda x: round(x,1), s.tolist()))
    b_r = list(map(lambda x: round(x,1), b.tolist()))
    plt.title('CrossEntropyLoss with 3 parallel networks, scale:%s, bias:%s' % (s_r, b_r))
    plt.show()
plot_results(best_cv, ind=True)

plt.figure()
plt.plot(loss)
plt.show()
