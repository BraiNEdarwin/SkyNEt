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
from pathlib import Path

from SkyNEt.modules.Nets.staNNet import staNNet
from SkyNEt.modules.Nets.webNNet import webNNet
from SkyNEt.instruments import InstrumentImporter
from SkyNEt.modules.SaveLib import createSaveDirectory, saveExperiment

class MeasureNet:
    """ Alternative class of staNNet to measure input data of a single device """
    
    def __init__(self, voltage_max, voltage_min, device='random', D_in=7, set_frequency=1000, amplification=100):
        self.D_in = 7
        self.voltage_max = voltage_max
        self.voltage_min= voltage_min
        self.set_frequency=set_frequency
        self.amplification = amplification
        if device=='cdaq':
            self.measure = self.cdaq
        else:
            print("WARNING: using randomly generated data, NOT measuring anything")
            self.measure = self.random
    
    def outputs(self, input_data, grad=True):
        """
        input_data      torch tensor (N, 7), control voltages to set
        returns         torch tensor (N, 1), measured output current in nA of device
        """
        input_data = input_data.numpy()
        output_data = self.measure(input_data)
        return torch.FloatTensor(output_data)
    
    def check_inputs(self, input_data):
        for i in range(input_data.shape[1]):
            over = input_data[:,i] > self.voltage_max[i]
            under = input_data[:,i] < self.voltage_min[i]
            if np.sum(over)>0 or np.sum(under)>0:
                print('WARN: input range exceeded, clipping input voltages')
            input_data[over,i] = self.voltage_max[i]
            input_data[under,i] = self.voltage_min[i]
        return input_data
    
    def cdaq(self, input_data):
        input_data = self.check_inputs(input_data)
        return InstrumentImporter.nidaqIO.IO_cDAQ(input_data.T, self.set_frequency).T*self.amplification
    
    def random(self, input_data):
        input_data = self.check_inputs(input_data)
        return np.mean(input_data, axis=1, keepdims=True) + np.random.rand(input_data.shape[0], 1)/10



# ---------------------------- START configure ---------------------------- #

    

# test script without measuring: random, use cdaq: cdaq
execute = 'cdaq'
amplification = 100 # on ivvi rack

save_location = r'D:/data/lennart/web_validation/'
save_name = 'boolean_v2_2-1_random'


# voltage bounds which should never be exceeded: (checked in MeasureNet)
voltage_max = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.3, 0.3])
voltage_min = np.array([-1.2, -1.2, -1.2, -1.2, -1.2, -0.7, -0.7])

# input voltages of boolean inputs (on/upper, off/lower)
input_lower = -1.2
input_upper = 0.6

N = 100 # number of data points of one of four input cases of boolean logic

set_frequency = 1000 # Hz
ramp_speed = 50 # V/s


# load device simulation
main_dir = r'C:/Users/PNPNteam/Documents/GitHub/pytorch_models/'
data_dir = 'MSE_d5w90_500ep_lr1e-3_b2048_b1b2_0.90.75-11-05-21h48m.pt'
net = staNNet(main_dir+data_dir)


mnet = MeasureNet(voltage_max, voltage_min, device=execute, set_frequency=set_frequency, amplification=amplification)

web = webNNet()
mweb = webNNet()

# define web for both NN model and measuring
for w,n in zip([web, mweb], [net, mnet]):
#    w.add_vertex(n, 'A', output=True, input_gates=[1,2])
    w.add_vertex(n, 'A', output=True, input_gates=[])
    w.add_vertex(n, 'B', input_gates=[1,2])
    w.add_vertex(n, 'C', input_gates=[1,2])
    w.add_arc('B', 'A', 1)
    w.add_arc('C', 'A', 2)



gates = ['AND', 'NAND', 'OR', 'NOR', 'XOR', 'XNOR']
#bool_cvs = [{'A': tensor([-1.1825,  0.2760, -0.3523,  0.2719, -0.4826])},
# {'A': tensor([ 0.2996, -0.8808, -0.5958, -0.2585,  0.1529])},
# {'A': tensor([-1.1245,  0.5861, -0.2215, -0.5269, -0.3841])},
# {'A': tensor([0.5000, 0.5838, 0.4203, 0.0130, 0.2996])},
# {'A': tensor([ 0.0724, -0.5670,  0.0325, -0.5061,  0.0676])},
# {'A': tensor([-1.1441,  0.1646, -0.7877,  0.3000, -0.2061])}]

bool_cvs = [{'A': tensor([ 0.2498, -0.3319, -1.0923, -0.1691, -1.1776, -0.0609,  0.0169]),
  'B': tensor([-1.0331, -0.2191, -1.1111, -0.1877, -0.6405]),
  'C': tensor([-0.2440, -0.8852,  0.0559,  0.0966,  0.2144])},
 {'A': tensor([ 0.0937,  0.3562, -0.8781, -1.1184,  0.0669, -0.6141,  0.0881]),
  'B': tensor([-0.7568,  0.3931,  0.1475,  0.2175,  0.0151]),
  'C': tensor([-0.7577, -0.3734, -0.2696, -0.5682,  0.2370])},
 {'A': tensor([-0.2924,  0.1317,  0.0832, -0.0645, -0.2500,  0.1185, -0.2103]),
  'B': tensor([-1.0824, -0.8645, -0.3586, -0.6862,  0.2656]),
  'C': tensor([-0.2053, -0.4878,  0.4077,  0.0027,  0.2233])},
 {'A': tensor([-0.5843,  0.4526, -0.5074, -0.1753,  0.2017, -0.1571,  0.1874]),
  'B': tensor([-1.0227, -0.9396, -0.0977, -0.4615, -0.2419]),
  'C': tensor([ 0.3603, -0.0894, -0.4416,  0.1514, -0.0895])},
 {'A': tensor([-0.9051, -0.7379, -0.5636, -0.3715, -0.8637, -0.6866,  0.1454]),
  'B': tensor([ 0.1546,  0.5784, -0.9241, -0.0073, -0.4881]),
  'C': tensor([ 0.2132, -0.0492, -0.9695, -0.4396, -0.0183])},
 {'A': tensor([ 0.0177, -0.3942, -0.6297, -1.1114,  0.3203, -0.2144,  0.1504]),
  'B': tensor([ 0.2061,  0.5553, -0.9211,  0.0597, -0.4676]),
  'C': tensor([ 0.5525,  0.2775, -0.8929, -0.3988, -0.3507])}]


# ---------------------------- END configure ---------------------------- #

savedir = createSaveDirectory(save_location, save_name)

def npramp(start, stop, set_frequency=1000, ramp_speed=50):
    """ ramp from start array to stop array, numpy version """
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
    """ ramp from start array to stop array, pytorch version """
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

model_output = {}
device_output = {}
model_alloutputs = {}
device_alloutputs = {}
for i, gate in enumerate(gates):
    cv = bool_cvs[i]
    keys = cv.keys()
    
    assert keys==web.graph.keys(), 'keys not matching'
    
    model_alloutputs[gate] = {}
    device_alloutputs[gate] = {}
    # get predicted output with neural network
    with torch.no_grad():
        web.reset_parameters(cv)
        web_output = web.forward(input_data)
    model_output[gate] = web_output
    
    # measure on device
    with torch.no_grad():
        mweb.reset_parameters(cv)
        mweb_output = mweb.forward(input_data)
    
    device_output[gate] = mweb_output

    # store all vertex outputs
    for key in keys:
        model_alloutputs[gate][key] = web.graph[key]['output'].numpy()
        device_alloutputs[gate][key] = mweb.graph[key]['output'].numpy()



saveExperiment(savedir, 
               input_data=input_data,
               control_voltages=bool_cvs,
               model_output=model_output,
               device_output=device_output,
               model_alloutputs=model_alloutputs,
               device_alloutputs=device_alloutputs)


InstrumentImporter.reset(0, 0)
