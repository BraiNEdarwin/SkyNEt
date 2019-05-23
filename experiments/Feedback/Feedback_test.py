# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:46:36 2019

@author: Jardi
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from SkyNEt.modules.Nets.staNNet import staNNet
from SkyNEt.modules.Nets.resNNet import resNNet

skip = 200
nodes = 100

## Load neural net
main_dir = r'C:/Users/Jardi/Desktop/BachelorOpdracht/NNModel/'
data_dir = '24-04-21h48m_NN_lossMSE-d20w90-lr0.003-eps500-mb2048-b10.9-b20.75.pt'
net = staNNet(main_dir+data_dir)

result_dir = r'../../../Resultaten/Delay line/'

## Load input signals
u = np.load(result_dir+'input.npy')
inpt = np.load(result_dir+'input_sampled.npy')
inpt_masked = np.load(result_dir+'input_masked.npy')

## Load output
output = np.load(result_dir+'output_inpt6_fdbck6_tau200_N2200.npy')
virout = np.load(result_dir+'output_inpt6_fdbck6_tau200_N2200_virtual.npy')

## Initiliasie reservoir
res = resNNet()
res.output = torch.from_numpy(virout)

## Train weights
weights, target = res.train_weights(u, nodes, skip)

prediction = np.dot(weights, np.transpose(virout))


x = np.linspace(1, len(output), len(output))
plt.figure()
plt.plot(x, inpt)
plt.plot(x, np.repeat(prediction[0,:], weights.shape[1]))
plt.plot(x, np.repeat(target[:,0], weights.shape[1]))

MCk = np.full(100, np.nan)
for i in range(nodes):
    MCk[i] = np.corrcoef(target[:,i], prediction[i,:])[0,1]**2
MC = sum(MCk)