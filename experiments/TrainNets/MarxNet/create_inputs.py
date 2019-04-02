# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:02:14 2019

@author: HCRuiz
"""

import numpy as np
from SkyNEt.modules.Nets.staNNet import staNNet as NN
import matplotlib.pyplot as plt

directory = r'D:\UTWENTE\PROJECTS\DARWIN\Data\Mark\\'
data_file = r'data_for_training_skip3.npz'

with np.load(directory+data_file) as data:
    print(list(data.keys()))
    outputs = data['outputs']
    inputs_indx = data['inputs']
    var_out = data['var_output']
    
net = NN(directory+'NN_skip3_MSE.pt')

freq = net.info['freq']
amplitude = net.info['amplitude']
offset = net.info['offset']
fs = net.info['fs']
phase = net.info['phase']

indices = np.arange(inputs_indx.max()+1)

sine_waves = amplitude*np.sin((2*np.pi*indices[:,np.newaxis]*freq + phase)/fs) + offset #Tel mark phase should be outside the brackets?

#plt.plot(sine_waves[:500])

inputs = sine_waves[inputs_indx[:,0]]

plt.plot(inputs[:1000])

np.savez(directory+'data_with_inputs.npz',outputs=outputs,inputs=inputs)
#DataLoader assumes data has keys 'outputs' and 'inputs'