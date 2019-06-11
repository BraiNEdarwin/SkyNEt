#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:36:08 2019

@author: lennart
"""

import numpy as np
import torch
import torch.tensor as tensor
import matplotlib.pyplot as plt

from SkyNEt.modules.Nets.staNNet import staNNet
from SkyNEt.modules.Nets.webNNet import webNNet
from SkyNEt.instruments import InstrumentImporter

class MeasureNet:
    """ Alternative class of staNNet to measure input data of a single device """
    
    def __init__(self, device='cdaq', D_in=7, set_frequency=1000):
        self.D_in = 7
        self.set_frequency=set_frequency
        if device=='cdaq':
            self.measure = self.cdaq
        else:
            assert False, 'Incorrect measurement device'
    
    def model(self, input_data):
        """
        input_data      torch tensor (N, 7), control voltages to set
        returns         torch tensor (N, 1), measured output current in nA of device
        """
        input_data = input_data.numpy()
        output_data = self.measure(input_data)
        return torch.FloatTensor(output_data)
    
    def cdaq(self, input_data):
#        return InstrumentImporter.nidaqIO.IO_cDAQ(input_data, self.set_frequency)
        return np.mean(input_data, axis=1, keepdims=True) + np.random.rand(input_data.shape[0], 1)/10



# input voltages of boolean inputs (on/upper, off/lower)
input_lower = -0.7
input_upper = 0.1

N = 100 # number of data points of one of four input cases of boolean logic

set_frequency = 1000 # Hz
ramp_speed = 50 # V/s

# load device simulation
main_dir = r'C:/Users/PNPNteam/Documents/GitHub/pytorch_models/'
data_dir = 'MSE_d5w90_500ep_lr3e-3_b2048_largeRange.pt'
net = staNNet(main_dir+data_dir)

mnet = MeasureNet(set_frequency=set_frequency)

web = webNNet()
mweb = webNNet()

# define web for both NN model and measuring
for w,n in zip([web, mweb], [net, mnet]):
    w.add_vertex(n, 'A', output=True, input_gates=[])
    w.add_vertex(n, 'B', input_gates=[4,5])
    w.add_vertex(n, 'C', input_gates=[4,5])
    w.add_arc('B', 'A', 4)
    w.add_arc('C', 'A', 5)

cv = {'A': tensor([-1.1410,  0.0939, -1.0543,  0.4253,  0.3122]), 'scale': tensor([ 1.8859, -3.0174]), 'bias': tensor([-23.1769,  -5.6604])}

keys = cv.keys()

assert keys==web.graph.keys(), 'keys not matching'

def npramp(start, stop, set_frequency=1000, ramp_speed=50):
    """ ramp from start array to stop array """
    delta = stop - start
    max_delta = max(abs(delta))
    # round up division number of steps are needed
    num = -(-max_delta*set_frequency//ramp_speed)
    if num<=1:
        return np.stack((start, stop))
    # calculate step size
    step = delta / num
    y = np.arange(0., num+1)
    y = np.outer(y, step)
    y += start
    return y

def ramp(start, stop, set_frequency=1000, ramp_speed=50):
    """ ramp from start array to stop array """
    delta = stop - start
    max_delta = max(abs(delta))
    # round up division number of steps are needed
    num = -(-max_delta*set_frequency//ramp_speed)
    if num<=1:
        return torch.stack((start, stop))
    # calculate step size
    step = delta / num
    y = torch.arange(0., num+1)
    y = torch.ger(y, step)
    y += start
    return y

# input data for both I0 and I1
problem_input_size = 2 # two binary inputs
case1 = torch.tensor([input_lower, input_lower])
case2 = torch.tensor([input_lower, input_upper])
case3 = torch.tensor([input_upper, input_lower])
case4 = torch.tensor([input_upper, input_upper])
# input ramping before I(0,0) and after I(1,1)
preramp = ramp(torch.zeros(problem_input_size), case1)
postramp = ramp(case4, torch.zeros(problem_input_size))

input_data = torch.cat((
        preramp,
        case1.repeat(N,1),
        ramp(case1,case2),
        case2.repeat(N,1),
        ramp(case2,case3),
        case3.repeat(N,1),
        ramp(case3,case4),
        case4.repeat(N,1),
        postramp))


# copy input for each network
stack_size = (7*len(web.graph) - web.nr_of_params)//problem_input_size
input_data = torch.cat((input_data,)*stack_size, dim=1)

# get predicted output with neural network
with torch.no_grad():
    web.reset_parameters(cv)
    web_output = web.forward(input_data)

# measure on device
with torch.no_grad():
    mweb_output = mweb.forward(input_data)

np_web_output = {}
np_mweb_output = {}
for key in keys:
    np_web_output[key] = web.graph[key]['output'].numpy()
    np_mweb_output[key] = mweb.graph[key]['output'].numpy()
    
    plt.figure()
    plt.plot(np_web_output[key])
    plt.plot(np_mweb_output[key])
    plt.legend(['web prediction', 'measured'])
    plt.title('Output of vertex {}'.format(key))
    plt.ylabel('output current (nA)')
    plt.show()

InstrumentImporter.reset(0, 0)
