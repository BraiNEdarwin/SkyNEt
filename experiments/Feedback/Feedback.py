# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:56:38 2019

@author: Jardi
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from SkyNEt.modules.Nets.staNNet import staNNet
from SkyNEt.modules.Nets.resNNet import resNNet

def Transferfunction(x):
    return (torch.clamp(x + 30, 0, 60))/60

# Constants and parameters
N = 1000
vlow1, vhigh1 = -0.8, 0.2
vlow2, vhigh2 = -1.1, 0.7
voltage_bounds = np.repeat([[vlow1, vlow2], [vhigh1, vhigh2]], [2, 5, 2, 5]).reshape(-1, 7).astype(np.float32)
input_electrode = 6

## input signal
inpt = torch.repeat_interleave(torch.FloatTensor(int(N/5), 1).uniform_(vlow2, vhigh2), 5).view(N, 1)
inpt_np = inpt.numpy()

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
res.add_vertex(net, 'A', output = True, input_gates = [input_electrode], voltage_bounds = voltage_bounds)
#res.add_vertex(net, 'B', output = True, input_gates = [], voltage_bounds = voltage_bounds)
#res.add_arc('A', 'B', input_electrode)
res.add_feedback('A', 'A', 0)

## forward pass
output = res.forward(inpt)
output_np = output.detach().numpy()

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(inpt.numpy())
plt.xlabel('Time step')
plt.ylabel('Input voltage (V)')
plt.title('Input (electrode 6)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(output.detach().numpy())
plt.xlabel('Time step')
plt.ylabel('Output current (nA)')
plt.title('Output')
plt.grid(True)
plt.tight_layout()

