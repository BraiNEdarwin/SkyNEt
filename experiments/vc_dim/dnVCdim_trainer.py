# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 18:26:56 2019
Train NN to perform binary classification on the input provided.
---------------
Arguments
inputs (list of lists) : 2d inputs providing the points of the data to be classified.
                         The index of the list containing the values corresponds to the dimension 
binary_labels (list)   : Targets 
filepath (kwarg, str)  : Path to save the results of the training 
---------------
Returns:
weights (np.array) : array containing weights of the net
output (np.array)  : array with the output of the net
cost (np.array)    : array with the costs (train) per epoch
accuracy (float)   : accuray of the percepron labelling the output of the net

Notes:
    The net is composed by a single hidden layer and an output unit returning a float.
    The output of the net is evaluated with the same cost function used for the device and
    it is evaluated for separability using a perceptron.

@author: hansr
"""

# SkyNEt imports
#import SkyNEt.modules.SaveLib as SaveLib
from SkyNEt.modules.Classifiers import perceptron
from SkyNEt.config.acceleration import Accelerator
from SkyNEt.modules.GenWaveform import GenWaveform
# Other imports
import torch
#import time
import numpy as np
#import pdb

#%% Define loss function
def neg_sig_corr(output, target):

    vx = output - torch.mean(output)
    vy = target - torch.mean(target)
    c = torch.sum(vx * vy)
    corr = c / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    
    x0 = output[target==target.min()]
    x1 = output[target==target.max()]
    x0 = x0[x0==torch.max(x0)]
    x1 = x1[x1==torch.min(x1)]
    dx = torch.mean(x1) - torch.mean(x0)
    f = 1.0/(1.0+torch.exp(-dx))
    print(f'corr: {corr}; sig: {f}')
    return (1.0-corr)/(f + 1e-7)

def input_waveform(inputs,lengths):
    assert len(inputs) == 2, 'Input must be 2 dimensional!'
    inp_wvfrm0 = GenWaveform(inputs[0], lengths)
    inp_wvfrm1 = GenWaveform(inputs[1], lengths)
    inputs_wvfrm = np.asarray([inp_wvfrm0,inp_wvfrm1],dtype=np.float32)
    
    return inputs_wvfrm

#%% Function definition
def train(inputs, binary_labels, net, loss_fn,
          filepath = r'../../test/NN_test/nnVCdim_testing/',
          epochs = 1000, learning_rate=0.03, beta=5):
    
    cost = np.zeros(epochs)
    
    #Generate and process data
    length = [100]
    input_wfrm = input_waveform(inputs,length).T
    input_wfrm = Accelerator.format_numpy(input_wfrm)
    targets = GenWaveform(binary_labels,length)
    targets = np.asarray(targets)
    targets = Accelerator.format_numpy(targets).view(-1,1)
    
    #Define optimizer
    optim = torch.optim.Adam(net.parameters(),lr=learning_rate)
    #Train net
    c_min = np.inf
    for ep in range(epochs): #need to use minibatch for such a small network?
        
        out = net(input_wfrm)
        loss = loss_fn(out,targets) + beta*net.regularizer()

        optim.zero_grad()
        loss.backward()
        optim.step()
        # Save training cost
        cost[ep] = loss.data.cpu().numpy()
        if cost[ep]<c_min:
            best_output = out.data.cpu().numpy()
            c_min = cost[ep]
            params = list(filter(lambda p: p.requires_grad, net.parameters()))
            weights = np.asarray([p.data.cpu().numpy() for p in params])
    
    targets = targets.data.cpu().numpy()
    accuracy, _, _ = perceptron(best_output,targets)
    return weights, best_output, cost, accuracy, targets

#%% Initialization
if __name__=='__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import matplotlib.pyplot as plt
    from SkyNEt.modules.Nets.dopantNet import dopantNet as NN
    #NN parameters
    nn_params = [1,2] # list with data input indices
    #Initialize net
    net = NN(nn_params)
    #Define loss function 
    loss_fn  = neg_sig_corr
    
    inputs = [[-0.7,0.7,-0.7,0.7],[-0.7,-0.7,0.7,0.7]]
#    [[-0.7,0.7,-0.7,0.7,-0.35,0.35,0.,0.],[-0.7,-0.7,0.7,0.7,0.,0.,-1.0,1.0]]
    binary_labels = [0,1,1,0]
    best_weights, best_output, cost, accuracy, targets = train(inputs,
                                                               binary_labels,
                                                               net, loss_fn)
    
    print(f'accuracy: {accuracy}')
    print(best_weights)
    plt.figure()
    plt.plot(cost)
    plt.show()
    
    plt.figure()
    plt.plot(best_output)
    plt.plot(targets)
    plt.show()