# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:44:47 2019

@author: Jardi
"""
##TODO: update fitness function
import torch
import numpy as np
import SkyNEt.experiments.Feedback.config_feedback_GA as config
from SkyNEt.modules.Nets.staNNet import staNNet
from SkyNEt.modules.Nets.resNNet import resNNet, Transferfunction
import matplotlib.pyplot as plt

cf = config.config_feedback_GA()
## Parameters
cf.vir_nodes = 10
cf.input_electrode = 6
cf.skip = 200
cf.output_nodes = 50
cf.generations = 30
cf.mutationrate = 0.1
cf.fitnessavg = 1
output = False

if cf.input_electrode == 5 or cf.input_electrode == 6:
    cf.generange = [[-1.2,1.2], [-1.2,1.2], [-1.2,1.2], [-1.2,1.2], [-1,1], [-1,1], [-5, 5], [-15, 15], [0, 2], [-5, 5]]

N = 1250
vlow1, vhigh1 = -1, 1
vlow2, vhigh2 = -1.2, 1.2
voltage_bounds = np.repeat([[vlow2, vlow1], [vhigh2, vhigh1]], [5, 2, 5, 2]).reshape(-1, 7).astype(np.float32)
input_bounds = torch.tensor(voltage_bounds[:, cf.input_electrode])

u = torch.FloatTensor(int(N*cf.fitnessavg), 1).uniform_(vlow2, vhigh2)
inpt = torch.repeat_interleave(u, cf.vir_nodes).view(cf.vir_nodes*N*cf.fitnessavg, 1)

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
geneArray, fitnessArray, memoryArray = res.trainGA(u, inpt, cf, True, output)

for i, val in enumerate(cf.generange):
    geneArray[:, :, i] = (val[1]-val[0])*geneArray[:, :, i] + val[0]
save_dir = r"D:/Jardi/Resultaten/"
np.save(save_dir + "geneArray_Vir" + str(cf.vir_nodes) + "_Out" + str(cf.output_nodes) + "_avg" + str(cf.fitnessavg) + "_gain" + "_generations" + str(cf.generations) + "_inpt" + str(cf.input_electrode), geneArray)
np.save(save_dir + "fitnessArray_Vir" + str(cf.vir_nodes) + "_Out" + str(cf.output_nodes) + "_avg" + str(cf.fitnessavg) + "_gain" + "_generations" + str(cf.generations) + "_inpt" + str(cf.input_electrode), fitnessArray)
np.save(save_dir + "memoryArray_Vir" + str(cf.vir_nodes) + "_Out" + str(cf.output_nodes) + "_avg" + str(cf.fitnessavg) + "_gain" + "_generations" + str(cf.generations) + "_inpt" + str(cf.input_electrode), memoryArray)

## Plot best fitness
indices = np.where(fitnessArray == np.amax(fitnessArray))
#out = outputArray[indices[0], indices[1], :]
#virout = res.get_virtual_outputs(cf.vir_nodes, out)
#weights, target = res.train_weights(u.numpy(), cf.output_nodes, cf.skip, virout)
#_, MCk = cf.getMC(virout, weights, target)

genes = geneArray[indices[0], indices[1], :]

## plot stuff
#plt.figure()
#x = np.linspace(1, cf.output_nodes, cf.output_nodes)
#plt.plot(x, memoryArray[indices[0], indices[1]].reshape((cf.output_nodes,)))
#plt.ylim([0,1.05])
#plt.title('Forgetting curve (D = ' + str(cf.vir_nodes) + ', n_max = ' + str(N - cf.output_nodes - cf.skip) + ')')
#plt.xlabel('i')
#plt.ylabel('Memory function m(i)')
#plt.grid(True)
#plt.tight_layout
