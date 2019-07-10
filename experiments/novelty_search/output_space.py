#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:29:42 2019

@author: ljknoll

two points are fed into the device, the output space is seen as 2D, one dimension for each data point
by changing the control voltages, we can map the same two input points to this 2D space
The question is can we reach the whole space with enough density?

In order to speed up computation, I consider my input data and control voltages swapped.
"""

import torch
from SkyNEt.modules.Nets.staNNet import staNNet
from SkyNEt.modules.Nets.webNNet import webNNet
import matplotlib.pyplot as plt


# number of random control voltages to check
n = 50000

points = torch.FloatTensor([[0,0],
                            [1,1]])

input_gates=[2,3,4,5,6]

# generate control voltage gates
control_gates = []
for i in range(7):
    if i not in input_gates:
        control_gates.append(i)

# prepare web
main_dir = r'/home/lennart/Dropbox/afstuderen/search_scripts/'
network_dir = 'NN_skip3_MSE.pt'
net1 = staNNet(main_dir+network_dir)

web = webNNet()
web.add_vertex(net1, 'A', output=True, input_gates=input_gates)


scaling = torch.FloatTensor(net1.info['amplitude']*2)
offset = torch.FloatTensor(net1.info['offset']) - scaling/2

# use voltage bounds for points
points = points*scaling[control_gates]+offset[control_gates]

# create random tensor of all voltages to be applied
cv = torch.rand(n,len(input_gates))*scaling[input_gates] + offset[input_gates]

# feed voltages through device for each of the points
output = torch.zeros(n, points.shape[0])
with torch.no_grad():
    for i,point in enumerate(points):
        web.reset_parameters(point)
        output[:,i] = web.forward(cv)[:,0]


def plot_output(output):
    plt.figure()
    plt.plot(output.numpy()[:,0], output.numpy()[:,1], ',k', alpha=0.6)
    margin = 0.05
    dx = (max(output[:,0])-min(output[:,0]))*margin
    dy = (max(output[:,1])-min(output[:,1]))*margin
    if max(output[:,0])>32:
        plt.plot([32,32], [min(output[:,1])-dy, max(output[:,1])+dy], 'k--')
    if min(output[:,0])<-32:
        plt.plot([-32,-32], [min(output[:,1])-dy, max(output[:,1])+dy], 'k--')
    if max(output[:,1])>32:
        plt.plot([min(output[:,0])-dx, max(output[:,0])+dx], [32,32], 'k--')
    if min(output[:,1])<-32:
        plt.plot([min(output[:,0])-dx, max(output[:,0])+dx], [-32,-32], 'k--')
    plt.xlabel('output (nA) when gate (Volt): %i (%0.1f), %i (%0.1f)' % (control_gates[0], points[0][0].item(), control_gates[1], points[0][1].item()))
    plt.ylabel('output (nA) when gate (Volt): %i (%0.1f), %i (%0.1f)' % (control_gates[0], points[1][0].item(), control_gates[1], points[1][1].item()))
    plt.title('output space characterization, random sampling')

plot_output(output)