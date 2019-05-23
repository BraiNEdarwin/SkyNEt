# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:56:38 2019

@author: Jardi
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from SkyNEt.modules.Nets.staNNet import staNNet
from SkyNEt.modules.Nets.resNNet import resNNet, Transferfunction

# Constants and parameters
N = 1000
vlow1, vhigh1 = -0.8, 0.2
vlow2, vhigh2 = -1.1, 0.7
voltage_bounds = np.repeat([[vlow1, vlow2], [vhigh1, vhigh2]], [2, 5, 2, 5]).reshape(-1, 7).astype(np.float32)
input_electrode = 6
feedback_electrode = 6
no_nodes = 4
skip = 0
runs = 10

## input signal
#inpt = torch.repeat_interleave(torch.FloatTensor(int(N/5), 1).uniform_(vlow2, vhigh2), 5).view(N, 1)
#torch.save(inpt, '../../../Resultaten/Chain met N naar 1 feedback/input.pt')
inpt = torch.load('../../../Resultaten/Chain met N naar 1 feedback/input.pt')
inpt_np = inpt.numpy()

## Load neural net
main_dir = r'C:/Users/Jardi/Desktop/BachelorOpdracht/NNModel/'
data_dir = '24-04-21h48m_NN_lossMSE-d20w90-lr0.003-eps500-mb2048-b10.9-b20.75.pt'
net = staNNet(main_dir+data_dir)

for run in range(runs):
    ## Initialise reservoir
    res = resNNet()
    
    d = res.graph
    
    ## Set transfer function
    #res.transfer = torch.nn.Hardtanh(0)
    res.transfer = Transferfunction

    ## Add devices
    #res.add_vertex(net, 'A', input_gates = [input_electrode], voltage_bounds = voltage_bounds)
    #res.add_vertex(net, 'B', output = True, input_gates = [], voltage_bounds = voltage_bounds)
    #res.add_arc('A', 'B', input_electrode)
    #res.add_feedback('B', 'A', input_electrode, input_bounds)
    res.init_chain(net, no_nodes, input_electrode, voltage_bounds, feedback_electrode, input_gain = 1, feedback_gain = 1)


    ## forward pass
    output = res.forward(inpt)
    output_np = output.detach().numpy()
    output_add = output[skip:,:].add(-output[skip:,:].min(0, keepdim=True)[0])
    torch.save(inpt, '../../../Resultaten/Chain met N naar 1 feedback/output_inpt' + str(input_electrode) + '_fback' + str(feedback_electrode) + '_nodes' + str(no_nodes) + '_skip' + str(skip) + '_run' + str(run+1) + '.pt')
    
    #compute trained weights
    weights, target = res.train_weights(inpt.numpy(), 3, skip)
    
    legends = []
    for i in range(no_nodes):
        string = 'Node ' + str(i+1)
        legends.append(string)
    
    plt.figure()
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(inpt.numpy()[skip:])
    plt.xlabel('Time step')
    plt.ylabel('Input voltage (V)')
    plt.title('Input (electrode ' + str(input_electrode) + ')')
    plt.grid(True)
    
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(output.detach().numpy()[skip:,no_nodes - 1])
    plt.xlabel('Time step')
    plt.ylabel('Output current (nA)')
    plt.title('Output of last node')
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig('../../../Resultaten/Chain met N naar 1 feedback/inout_inpt' + str(input_electrode) + '_fback' + str(feedback_electrode) + '_nodes' + str(no_nodes) + '_skip' + str(skip) + '_run' + str(run+1) + '.svg')
    
    plt.figure()
    plt.plot(output_add.detach().numpy())
    plt.xlabel('TIme step')
    plt.ylabel('Shifted output current')
    plt.title('Shifted output')
    plt.grid(True)
    plt.legend(legends, loc='best')
    plt.tight_layout()
    
    plt.savefig('../../../Resultaten/Chain met N naar 1 feedback/shiftout_inpt' + str(input_electrode) + '_fback' + str(feedback_electrode) + '_nodes' + str(no_nodes) + '_skip' + str(skip) + '_run' + str(run+1) + '.svg')