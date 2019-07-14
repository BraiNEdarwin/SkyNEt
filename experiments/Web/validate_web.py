#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:36:08 2019

Script which measures the device output physically instead of with a neural network model.
Instead of passing a staNNet object to add_vertex, MeasureNet object defined below is used.


@author: ljknoll
"""

import numpy as np
import torch
import torch.tensor as tensor
import matplotlib.pyplot as plt

from SkyNEt.modules.Nets.staNNet import staNNet
from SkyNEt.modules.Nets.webNNet import webNNet
from SkyNEt.instruments import InstrumentImporter

def npramp(start, stop, set_frequency=1000, ramp_speed=5):
    """ 2D version of linspace, ramp from start array to stop array using numpy"""
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

def ramp(start, stop, set_frequency=1000, ramp_speed=5):
    """ 2D version of linspace, ramp from start array to stop array using pytorch"""
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

class MeasureNet:
    """
    Alternative class of staNNet to measure output data of a single device
    Arguments:
        device          string, 
        D_in            integer, 
        set_frequency   integer, frequency used to 
    """
    
    def __init__(self, device='cdaq', D_in=7, set_frequency=1000):
        self.D_in = 7
        self.set_frequency=set_frequency
        if device=='cdaq':
            self.measure = self.cdaq
        elif device=='random':
            self.measure = self.random
            print('WARNING: using random outputs for testing, NOT measuring!')
        else:
            assert False, 'Incorrect measurement device'
    
    def outputs(self, input_data, grad=True):
        """
        input_data      torch tensor (N, 7), control voltages to set
        returns         torch tensor (N, 1), measured output current in nA of device
        """
        input_data = input_data.numpy()
        output_data = self.measure(input_data)
        return torch.FloatTensor(output_data)
    
    def cdaq(self, input_data):
        return InstrumentImporter.nidaqIO.IO_cDAQ(input_data, self.set_frequency)
    
    def random(self, input_data):
        return np.mean(input_data, axis=1, keepdims=True) + np.random.rand(input_data.shape[0], 1)/10

# ------------------ START CONFIGURE ------------------ #

# input voltages of boolean inputs (on/upper, off/lower)
input_lower = -0.7
input_upper = 0.1

N = 100 # number of data points of one of four input cases of boolean logic

set_frequency = 1000 # Hz
ramp_speed = 5 # V/s

device = 'random' # use 'cdaq'

# load device simulation
main_dir = r'/home/lennart/Dropbox/afstuderen/search_scripts/'
data_dir = 'NN_skip3_MSE.pt'
net = staNNet(main_dir+data_dir)

mnet = MeasureNet(device=device, D_in=7, set_frequency=set_frequency)

web = webNNet()
mweb = webNNet()

# define web for both NN model and measuring
for w,n in zip([web, mweb], [net, mnet]):
    w.add_vertex(n, 'A', output=True, input_gates=[])
    w.add_vertex(n, 'B', input_gates=[4,5])
    w.add_vertex(n, 'C', input_gates=[4,5])
    w.add_arc('B', 'A', 4)
    w.add_arc('C', 'A', 5)


cv = {'A': tensor([ 0.1174, -0.5477,  0.3110,  0.1646,  0.4120,  0.5261,  0.5301]),
  'B': tensor([ 0.1508, -0.0177, -0.2225,  0.0073, -0.2982]),
  'C': tensor([-0.3451, -0.4791,  0.4497, -0.5789, -0.0036])}


# ------------------ END CONFIGURE ------------------ #

keys = cv.keys()

assert keys==web.graph.keys(), 'keys not matching'


# input data for both I0 and I1
problem_input_size = 2 # two binary inputs
case1 = torch.tensor([input_lower, input_lower])
case2 = torch.tensor([input_lower, input_upper])
case3 = torch.tensor([input_upper, input_lower])
case4 = torch.tensor([input_upper, input_upper])
# input ramping before I(0,0) and after I(1,1)
preramp = ramp(torch.zeros(problem_input_size), case1, ramp_speed)
postramp = ramp(case4, torch.zeros(problem_input_size), ramp_speed)

input_data = torch.cat((
        preramp,
        case1.repeat(N,1),
        ramp(case1,case2, ramp_speed),
        case2.repeat(N,1),
        ramp(case2,case3, ramp_speed),
        case3.repeat(N,1),
        ramp(case3,case4, ramp_speed),
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

# plot output of each vertex
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
