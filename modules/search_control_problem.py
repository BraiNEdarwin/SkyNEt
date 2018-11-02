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

data_dir = r'/home/lennart/Desktop/2018_11_02_142235_CP_inputs_and_targets.npz'
data = np.load(data_dir)
input_data = torch.tensor(data['inputs']).float()
target_data = torch.tensor(data['target']).float()

main_dir = r'/home/lennart/Desktop/nnweb/'
data_dir = 'lr2e-4_eps400_mb512_20180807CP.pt'
net1 = predNNet(main_dir+data_dir)

web = webNNet()
web.add_vertex(net1, 'A', output=True)
web.add_vertex(net1, 'B')
#web.add_vertex(net1, 'C')
#web.add_vertex(net1, 'D')
#web.add_vertex(net1, 'E')

web.add_arc('B', 'A', 5)
#web.add_arc('C', 'B', 2)
#web.add_arc('D', 'C', 2)
#web.add_arc('E', 'D', 2)


batch_size = 150
nr_epochs = 500
lr = 0.05
beta = 0.1
cv_reset = 'rand' # 0.6*torch.ones(5) # None, 'rand', tensor(5)

add_noise = False
sigma = 0.01 # standard deviation of noise in target


optimizer = torch.optim.Adam
betas=(0.9, 0.75)
eps=1e-08

loss_fn = torch.nn.MSELoss()

if add_noise:
    gauss = torch.distributions.Normal(0.0, sigma)
    target_data += gauss.sample(target_data.shape)

loss, best_cv = web.train(input_data, target_data, 
                     beta=beta,
                     batch_size=batch_size,
                     nr_epochs=nr_epochs,
                     optimizer=optimizer,
                     loss_fn=loss_fn,
                     lr = lr,
                     betas = betas,
                     eps = eps)

def plot_results(best_cv):
    web.reset_parameters(best_cv)
    output_data = web.forward(input_data).data
    plt.figure()
    plt.plot(target_data)
    plt.plot(output_data)
    plt.title('output_data vs target data')
    plt.show()

plt.figure()
plt.plot(loss)
plt.show()

plot_results(best_cv)