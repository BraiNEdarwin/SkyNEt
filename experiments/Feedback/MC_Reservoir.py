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
vir_nodes = 50
N = 10000
nodes = 50

## Input Signal
u = torch.FloatTensor(int(N), 1).uniform_(-0.6, 0.6)
#u = torch.load(r'C:\Users\Jardi\Desktop\BachelorOpdracht\Resultaten\Delay line\input.pt')
u_np = u.numpy()

## Initialise reservoir
res = Network((1, vir_nodes, nodes), 1, 0.98, 0)

## forward pass
for i, val in enumerate(u_np):
    res.update_reservoir(val.reshape(1,1))
    
target = np.full((N, nodes), np.nan)
for i in range(nodes):
    target[:,i] = np.roll(u_np, i+1).reshape(N,)
    
prediction = res.train_reservoir_ridgereg(target, 0.1, nodes)

MCk = np.full(nodes, np.nan)
for i in range(nodes):
    MCk[i] = np.corrcoef(target[nodes:,i], prediction[i,:])[0,1]**2
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

def validateMC(weights, it):
    MCk_val = np.zeros(nodes)
    MC_it = np.full(it, np.nan)
    for i in range(it):
        u_val = torch.FloatTensor(int(N), 1).uniform_(-0.6, 0.6)

        res = Network((1, vir_nodes, nodes), 1, 0.98, 0)
        for ii, val in enumerate(u_val):
            res.update_reservoir(val.reshape(1,1))
        
        target_val = np.full((N, nodes), np.nan)
        for ii in range(nodes):
            target_val[:,ii] = np.roll(u_val, ii+1).reshape(N,)
            
        prediction = np.dot(weights, res.collect_state[nodes:].transpose())
        
        MCk_it = np.full(nodes, np.nan)
        for ii in range(nodes):
            MCk_it[ii] = np.corrcoef(target_val[nodes:,ii], prediction[ii,:])[0,1]**2
        MCk_val += MCk_it
        MC_it[i] = sum(MCk_it)
        
        plt.figure()
        x = np.linspace(1, nodes, nodes)
        plt.plot(x, MCk_it)
        plt.ylim([0,1.05])
        plt.title('Forgetting curve (D = ' + str(vir_nodes) + ', n_max = ' + str(N - nodes) + ')')
        plt.xlabel('i')
        plt.ylabel('Memory function m(i)')
        plt.grid(True)
        plt.tight_layout
    
    MCk_val /= it
    MC_val = sum(MCk_val)
    return MC_val, MCk_val, MC_it

MC_val, MCk_val, MC_it = validateMC(res.trained_weights, 3)