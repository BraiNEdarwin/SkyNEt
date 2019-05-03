# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:21:06 2019

@author: Jardi
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from SkyNEt.modules.Nets.staNNet import staNNet

## constants
N = 1000
vlow1 = -0.8
vhigh1 = 0.2
vlow2 = -1.1
vhigh2 = 0.7
feedback_electrode = 2
init_value = 0
delay = 5

## input/control voltages
inpt1 = np.linspace(vlow1, vhigh1, num=N)
inpt2 = np.linspace(vlow2, vhigh2, num=N)
inpt3 = np.concatenate((np.full(200, vlow2), np.full(200, vhigh2), np.full(200, vlow2), np.full(100, vhigh2), np.full(300, vlow2)))
e0 = np.ones(N) * 0
e1 = np.ones(N) * 0
e2 = np.ones(N) * init_value
e3 = np.ones(N) * 0.4
e4 = np.ones(N) * 0.5
e5 = inpt2
e6 = inpt3

## Create input matrix
#inputs = torch.FloatTensor(N, net.D_in).uniform_(vlow, vhigh)
#inputs_numpy = inputs.numpy()
inputs_numpy =  np.column_stack((e0, e1, e2, e3, e4, e5, e6))
inputs = torch.from_numpy(inputs_numpy).float()

## Load neural net
main_dir = r'C:/Users/Jardi/Desktop/BachelorOpdracht/NNModel/'
data_dir = '24-04-21h48m_NN_lossMSE-d20w90-lr0.003-eps500-mb2048-b10.9-b20.75.pt'
net = staNNet(main_dir+data_dir)

## predict output without feedback
prediction1 = net.outputs(inputs)

## add feedback
net.add_feedback(feedback_electrode, init_value, delay)

## predict output with feedback
prediction2, inputs_feedback = net.outputs(inputs)

## plot stuff
plt.subplot(2, 1, 1)
plt.plot(inpt3)
plt.xlabel('Time step')
plt.ylabel('Input voltage (V)')
plt.title('Input (electrode 6)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(prediction1)
plt.plot(prediction2)
plt.xlabel('Time step')
plt.ylabel('Output current (nA)')
plt.title('Output')
plt.grid(True)
plt.legend(('Without feedback', 'With feedback'), loc='best')

plt.tight_layout()