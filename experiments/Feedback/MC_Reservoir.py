# -*- coding: utf-8 -*-
"""
Created on Tue May 21 12:32:47 2019

@author: Jardi
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from SkyNEt.modules.ReservoirFull import Network

## Parameters
tau = 400
N = 10000
nodes = 100

## Input Signal
u = torch.FloatTensor(int(N), 1).uniform_(-0.6, 0.6)
#u = torch.load(r'C:\Users\Jardi\Desktop\BachelorOpdracht\Resultaten\Delay line\input.pt')
u_np = u.numpy()

## Initialise reservoir
res = Network((1, tau, nodes), 1, 0.98, 0)

## forward pass
for i, val in enumerate(u_np):
    res.update_reservoir(val.reshape(1,1))
    
target = np.full((N, nodes), np.nan)
for i in range(nodes):
    target[:,i] = np.roll(u_np, i+1).reshape(N,)
    
prediction = res.train_reservoir_pseudoinv(target, nodes)

MCk = np.full(nodes, np.nan)
for i in range(nodes):
    MCk[i] = np.corrcoef(target[100:,i], prediction[i,:])[0,1]**2
MC = sum(MCk)

## plot stuff
plt.figure()
x = np.linspace(1, nodes, nodes)
plt.plot(x, MCk)
plt.ylim([0,1.05])
plt.title('Forgetting curve')
plt.xlabel('i')
plt.ylabel('Memory function m(i)')
plt.grid(True)
plt.tight_layout