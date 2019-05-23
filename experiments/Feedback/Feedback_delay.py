# -*- coding: utf-8 -*-
"""
Created on Tue May 21 12:32:47 2019

@author: Jardi
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from SkyNEt.modules.Nets.staNNet import staNNet
from SkyNEt.modules.Nets.resNNet import resNNet, Transferfunction


## Parameters
tau = 400
N = 1300
vlow1, vhigh1 = -0.8, 0.2
vlow2, vhigh2 = -1.1, 0.7
voltage_bounds = np.repeat([[vlow1, vlow2], [vhigh1, vhigh2]], [2, 5, 2, 5]).reshape(-1, 7).astype(np.float32)
input_electrode = 6
feedback_electrode = 6
input_bounds = torch.tensor(voltage_bounds[:, feedback_electrode])
skip = 200
nodes = 100
input_gain = 1
feedback_gain = 1

## Input Signal
u = torch.FloatTensor(int(N), 1).uniform_(vlow2/1.25, vhigh2/1.25)
u = torch.load(r'C:\Users\Jardi\Desktop\BachelorOpdracht\Resultaten\Delay line\input.pt')
u_np = u.numpy()
inpt = torch.repeat_interleave(u, tau).view(tau*N, 1)
mask = torch.rand_like(inpt)/2 + 0.75
mask = torch.load(r'C:\Users\Jardi\Desktop\BachelorOpdracht\Resultaten\Delay line\mask.pt')
inpt_mask = inpt * mask
inpt_np = inpt.numpy()
inpt_mask_np = inpt_mask.numpy()

## Load neural net
main_dir = r'C:/Users/Jardi/Desktop/BachelorOpdracht/NNModel/'
data_dir = '24-04-21h48m_NN_lossMSE-d20w90-lr0.003-eps500-mb2048-b10.9-b20.75.pt'
net = staNNet(main_dir+data_dir)

## Initialise reservoir
res = resNNet()

d = res.graph

## Set transfer function
#res.transfer = torch.nn.Hardtanh(0)
res.transfer = Transferfunction

## Add devices
res.add_vertex(net, '0', output = True, input_gates = [input_electrode], voltage_bounds = voltage_bounds)
res.add_feedback('0', '0', feedback_electrode, input_bounds, input_gain, feedback_gain)

## forward pass
output = res.forward_delay(inpt_mask, tau)
output_np = output.detach().numpy()
virout = res.get_virtual_outputs(tau)
virout_np = virout.detach().numpy()

weights, target = res.train_weights(u_np, nodes, skip)

prediction = np.dot(weights, np.transpose(virout_np[skip+nodes:,:]))

MCk = np.full(nodes, np.nan)
for i in range(nodes):
    MCk[i] = np.corrcoef(target[:,i], prediction[i,:])[0,1]**2
MC = sum(MCk)

## plot stuff
plt.figure()
x = np.linspace(1, nodes, nodes)
plt.plot(x, MCk)


#plt.figure()
#ax1 = plt.subplot(3, 1, 1)
#ax1.plot(inpt_np[skip:skip+20*N])
#plt.xlabel('Time step')
#plt.ylabel('Input voltage (V)')
#plt.title('Input (electrode ' + str(input_electrode) + ')')
#plt.grid(True)
#
#ax2 = plt.subplot(3, 1, 2)
#ax2.plot(inpt_mask_np[skip:skip+20*N])
#plt.xlabel('Time step')
#plt.ylabel('Masked input voltage (V)')
#plt.title('Masked input')
#plt.grid(True)
#plt.tight_layout()
#
#ax3 = plt.subplot(3, 1, 3)
#ax3.plot(output_np[skip:skip+20*N])
#plt.xlabel('Time step')
#plt.ylabel('Output current (nA)')
#plt.title('Output current')
#plt.grid(True)
#plt.tight_layout()
#
#plt.figure()
#plt.plot(virout_np)
#plt.title('Virtual outputs')
#plt.grid(True)
#plt.tight_layout()
#
x = np.linspace(skip+nodes+1, tau*N, tau*(N-skip-nodes))
#
#plt.figure()
#plt.plot(x, inpt_np)
#plt.plot(x, np.transpose(np.repeat(prediction, tau, axis=1)))
#plt.title('Delayed outputs')
#plt.grid(True)
#plt.tight_layout()

plt.figure()
plt.plot(x, inpt_np[tau*(skip+nodes):])
plt.plot(x, np.repeat(prediction[0,:], tau))
plt.plot(x, np.repeat(target[:,0], tau))