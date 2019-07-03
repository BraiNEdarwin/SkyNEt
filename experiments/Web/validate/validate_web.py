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
        diff = (voltage_max-voltage_min)/2
        self.info = {'offset': diff+voltage_min, 'amplitude':diff}
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
            if np.sum(over)>0:
                print('WARN: max input range exceeded, clipping input voltages: %.3f' % np.max(input_data[over, i]))
            if np.sum(under)>0:
                print('WARN: min input range exceeded, clipping input voltages: %.3f' % np.min(input_data[under, i]))
            input_data[over,i] = self.voltage_max[i]
            input_data[under,i] = self.voltage_min[i]
            print('min: %.3f, max: %.3f' % (input_data[:,i].min(), input_data[:,i].max()))
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
#save_location = r'/home/lennart/Dropbox/afstuderen/results/measurements/'
save_name = 'cp_v2_4-1'

n=100 # datapoints per problem case
result_folder = '2019_06_17_142417_cp_GD_4-1_v2_beta80-92' # only saved with experiment, not used

# voltage bounds which should never be exceeded: (checked in MeasureNet)
voltage_max = np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.3, 0.3])
voltage_min = np.array([-1.2, -1.2, -1.2, -1.2, -1.2, -0.7, -0.7])

set_frequency = 1000 # Hz
ramp_speed = 0.1 # V/s, 
# IMPORTANT: ivvi rack ramps each voltage individually, so this value is only used for changing a single voltage at a time


# load device simulation
main_dir = r'C:/Users/PNPNteam/Documents/GitHub/pytorch_models/'
#main_dir = r'/home/lennart/Dropbox/afstuderen/search_scripts/'
data_dir = 'MSE_d5w90_500ep_lr1e-3_b2048_b1b2_0.90.75-11-05-21h48m.pt'
net = staNNet(main_dir+data_dir)

def cust_sig(x):
#    print('cust_sig')
    return torch.sigmoid(x/15*2)

mnet = MeasureNet(voltage_max, voltage_min, device=execute, set_frequency=set_frequency, amplification=amplification)

web = webNNet()
mweb = webNNet()
web.transfer = cust_sig
mweb.transfer = cust_sig

# define web for both NN model and measuring
for w,n in zip([web, mweb], [net, mnet]):
#    w.add_vertex(n, 'A', output=True, input_gates=[1,2])
    w.add_vertex(n, 'A', output=True, input_gates=[])
    w.add_vertex(n, 'B', input_gates=[1,2])
    w.add_vertex(n, 'C', input_gates=[1,2])
    w.add_vertex(n, 'D', input_gates=[1,2])
    w.add_vertex(n, 'E', input_gates=[1,2])
    w.add_arc('B', 'A', 1)
    w.add_arc('C', 'A', 2)
    w.add_arc('D', 'A', 3)
    w.add_arc('E', 'A', 4)
    w.add_parameters(['scale', 'bias'],
                       [torch.ones(2), torch.tensor([0., 1.])],
    #                   lambda : 0.0001*torch.mean(web.scale**2 + web.bias**2)/2,
                       lr=0.1, betas=(0.9,0.99))


cv_list = torch.tensor([-1.1044788360595703,
 -0.20852607488632202,
 -0.1164623498916626,
 -0.6689516305923462,
 -0.19157111644744873,
 -0.3318597674369812,
 0.11369849741458893,
 -0.4117467999458313,
 -0.975537896156311,
 -1.0046099424362183,
 -0.06706836819648743,
 -0.005668036639690399,
 0.12973833084106445,
 -1.1843717098236084,
 0.11103629320859909,
 -0.028120175004005432,
 0.2865794003009796,
 -1.1123498678207397,
 -0.9114303588867188,
 -0.9430482983589172,
 0.1317417174577713,
 -0.17312610149383545,
 -0.04973844438791275,
 -0.32583948969841003,
 0.15769539773464203,
 -0.31805670261383057,
 0.028909362852573395,
 4.800425052642822,
 -14.237703323364258,
 -5.183893203735352,
 -1.211928367614746])

web.reset_parameters(cv_list)
cv = web.get_parameters()


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


# function to generate train data for control problem
#def generate_cp(n=10, mean_I0=-0.2, mean_I1=-0.2, amp_I0=0.9, amp_I1=0.9):
#     values_I0 = [mean_I0-amp_I0+amp_I0*2/2*(i//n//7) for i in range(21*n)]
#     values_I1 = [mean_I1-amp_I1+amp_I1*2/6*(i//n%7) for i in range(21*n)]
#     input_data = torch.tensor([values_I0, values_I1]).t()
#     targets = [0,0,0,1,1,1,1,0,1,1,1,1,2,2,1,1,2,1,2,1,2]
#     target_data = torch.tensor([targets]).view(-1,1).repeat(1,n).view(-1,1)
#     return input_data, target_data
    
# input data for both I0 and I1
def generate_cp(n=10, mean_I0=-0.2, mean_I1=-0.2, amp_I0=0.9, amp_I1=0.9):
    N=1
    values_I0 = [mean_I0-amp_I0+amp_I0*2/2*(i//N//7) for i in range(21*N)]
    values_I1 = [mean_I1-amp_I1+amp_I1*2/6*(i//N%7) for i in range(21*N)]
    raw_problem_data = torch.tensor([values_I0, values_I1]).t()
    raw_input_data = torch.cat((torch.zeros(N,2), raw_problem_data, torch.zeros(N,2)))
    input_data_list = []
    positions = []
    count = 0
    for i in range(21):
        ramped_data = ramp(raw_input_data[i], raw_input_data[i+1])
        input_data_list.append(ramped_data)
        input_data_list.append(raw_input_data[i+1].repeat(n,1))
        count += ramped_data.shape[0]
        positions.append(count+n//2)
        count +=n
    input_data_list.append(ramp(raw_input_data[-2], raw_input_data[-1]))
    input_data = torch.cat(input_data_list)
    return input_data, positions
#    targets = [0,0,0,1,1,1,1,0,1,1,1,1,2,2,1,1,2,1,2,1,2]
#    target_data = torch.tensor([targets]).view(-1,1).repeat(1,n).view(-1,1)
#    return input_data, target_data
input_data, positions = generate_cp(n=100)


# copy input for each network
stack_size = (7*len(web.graph) - web.nr_of_params)//2
input_data = torch.cat((input_data,)*stack_size, dim=1)


keys = cv.keys()

assert keys==web.get_parameters().keys(), 'keys not matching'


model_alloutputs = {}
device_alloutputs = {}
print('start model')
# get predicted output with neural network
with torch.no_grad():
    web.reset_parameters(cv)
    web_output = web.forward(input_data)
model_output = web_output

print('start measuring')
# measure on device
with torch.no_grad():
    mweb.reset_parameters(cv)
    mweb_output = mweb.forward(input_data)

device_output = mweb_output

# store all vertex outputs
for key in web.graph.keys():
    model_alloutputs[key] = web.graph[key]['output'].numpy()
    device_alloutputs[key] = mweb.graph[key]['output'].numpy()



saveExperiment(savedir, 
               input_data=input_data,
               control_voltages=cv,
               model_output=model_output,
               device_output=device_output,
               model_alloutputs=model_alloutputs,
               device_alloutputs=device_alloutputs,
               positions=positions,
               result_folder=result_folder)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_output(device_output, model_output, positions, window=1, extra_title='', numpy=False):
    plt.figure()
    xticks = ['A','B','C','D','E','F','G']
    plt.xticks(positions, xticks*3)
    if numpy:
        plt.plot(moving_average(device_output, window), 'k')
        plt.plot(model_output, ':k')
    else:
        plt.plot(moving_average(device_output.numpy(), window), 'k')
        plt.plot(model_output.numpy(), ':k')
    plt.title(extra_title)
    plt.legend(['device', 'model'])
    plt.ylabel('Output current (nA)')
    plt.show()

window = 10

plot_output(device_output, model_output, positions)
plot_output(device_output, model_output, positions, window=window)

vertices = model_alloutputs.keys()
for vertex in vertices:
    plot_output(device_alloutputs[vertex], model_alloutputs[vertex], positions, extra_title='vertex '+vertex, numpy=True)
    plot_output(device_alloutputs[vertex], model_alloutputs[vertex], positions, extra_title='vertex '+vertex+', window '+str(window), window=10, numpy=True)

def plot_input_data(input_data, fontsize=25):
    plt.rcParams.update({'font.size':fontsize})
    xticks = ['A','B','C','D','E','F','G']
    plt.plot(input_data[:,0].numpy(), ':k')
    plt.plot(input_data[:,1].numpy(), 'k')
    plt.xticks(positions, xticks*3)
    plt.legend(['I0', 'I1'])
    plt.ylabel('Applied voltage (V)')
    plt.show()

InstrumentImporter.reset(0, 0)
