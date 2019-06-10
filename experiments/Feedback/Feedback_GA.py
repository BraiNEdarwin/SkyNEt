# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:44:47 2019

@author: Jardi
"""

import torch
import numpy as np
import SkyNEt.experiments.Feedback.config_feedback_GA as config
from SkyNEt.modules.Nets.staNNet import staNNet
from SkyNEt.modules.Nets.resNNet import resNNet, Transferfunction
import matplotlib.pyplot as plt

cf = config.config_feedback_GA()

## Parameters
cf.vir_nodes = 100
cf.theta = 1
cf.input_electrode = 3
cf.skip = 200
cf.output_nodes = 50
cf.generations = 10
cf.mutationrate = 0.1
output = False

N = 5250
vlow1, vhigh1 = -1, 1
vlow2, vhigh2 = -1.2, 1.2
voltage_bounds = np.repeat([[vlow2, vlow1], [vhigh2, vhigh1]], [5, 2, 5, 2]).reshape(-1, 7).astype(np.float32)
input_bounds = torch.tensor(voltage_bounds[:, cf.input_electrode])

u = torch.FloatTensor(int(N), 1).uniform_(vlow2, vhigh2)

## Load neural net
main_dir = r'D:/Jardi/NNModels/'
data_dir = 'MSE_n_d5w90_500ep_lr3e-3_b2048.pt'
net = staNNet(main_dir+data_dir)

## Initialise reservoir
res = resNNet()
res.transfer = Transferfunction

## Add devices
res.add_vertex(net, '0', output = True, input_gates = [cf.input_electrode], voltage_bounds = voltage_bounds)
res.add_feedback('0', '0', cf.input_electrode, input_bounds)

## Train with GA
geneArray, fitnessArray, memoryArray = res.trainGA(u, cf, True, output)
save_dir = r"D:/Jardi/Resultaten/"

## Plot best fitness
indices = np.where(fitnessArray == np.amax(fitnessArray))
#out = outputArray[indices[0], indices[1], :]
#virout = res.get_virtual_outputs(cf.vir_nodes, out)
#weights, target = res.train_weights(u.numpy(), cf.output_nodes, cf.skip, virout)
#_, MCk = cf.getMC(virout, weights, target)

genes = geneArray[indices[0], indices[1], :]

## plot stuff
plt.figure()
x = np.linspace(1, cf.output_nodes, cf.output_nodes)
plt.plot(x, memoryArray[indices[0], indices[1], :].reshape((50,)))
plt.ylim([0,1.05])
plt.title('Forgetting curve (D = ' + str(cf.vir_nodes) + ', n_max = ' + str(N - cf.output_nodes - cf.skip) + ')')
plt.xlabel('i')
plt.ylabel('Memory function m(i)')
plt.grid(True)
plt.tight_layout
