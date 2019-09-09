#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:33:38 2019

@author: hruiz
"""

from SkyNEt.modules.Nets.staNNet import staNNet
import numpy as np
import torch.nn as nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class dopantNet(nn.Module):

    def __init__(self,in_list,
		path=r'../../test/NN_test/checkpoint3000_02-07-23h47m.pt'):
        super(dopantNet,self).__init__()
        
        self.in_list = in_list
        self.nr_inputs = len(in_list)
        self.net = staNNet(path).to(device)
        #Freeze parameters
        for params in self.net.parameters():
            params.requires_grad = False
        # Define learning parameters
        self.nr_electodes = len(self.net.info['offset'])
        self.indx_cv = np.delete(np.arange(self.nr_electodes),in_list)
        self.nr_cv = len(self.indx_cv)
        offset = self.net.info['offset']
        amplitude =  self.net.info['amplitude']
        
        self.min_voltage = offset - amplitude
        self.max_voltage = offset + amplitude
        bias = self.min_voltage + (self.max_voltage - self.min_voltage)*np.random.rand(1,self.nr_cv)
        bias = torch.tensor(bias,dtype=torch.float32).to(device)
        self.bias = nn.Parameter(bias)
        #Set as torch Tensors and send to device
        self.indx_cv = torch.tensor(self.indx_cv).to(device)
        self.amplification = torch.tensor(self.net.info['amplification']).to(device)
    
    def forward(self,x):
        
        expand_cv = self.bias.expand(x.size()[0],-1)
        inp = torch.empty((x.size()[0],x.size()[1]+self.nr_cv)).to(device)
#        print(x.dtype,self.amplification.dtype)

        inp[:,self.in_list] = x
        
#        print(inp.dtype,self.indx_cv.dtype,expand_cv.dtype)
        inp[:,self.indx_cv] = expand_cv
        
        return self.net.model(inp)*self.amplification
    
    
    def regularizer(self):
        assert any(self.min_voltage[self.indx_cv]<0), \
        "Min. Voltage is assumed to be negative, but value is positive!"
        assert any(self.max_voltage[self.indx_cv]>0), \
        "Max. Voltage is assumed to be positive, but value is negative!"
        reg = torch.sum(torch.relu(self.min_voltage[self.indx_cv] - x) 
                        + torch.relu(x-self.max_voltage[self.indx_cv])) 
        return reg
        


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    x = 0.5*np.random.randn(1,3) 
    x = torch.Tensor(x).to(device)
    
    loss = nn.MSELoss()
    target = torch.Tensor([5]).to(device)
    
    node = dopantNet([0,3,4])
    print(list(node.parameters())[0])
    print(list(node.parameters())[-1][:8])

#    optimizer = torch.optim.SGD([{'params':filter(lambda p: p.requires_grad, node.parameters())}],lr=0.0001)
    optimizer = torch.optim.SGD([{'params':node.parameters()}],lr=0.0001)
    
    loss_array = []
    for i in range(200):
        
        optimizer.zero_grad()         
        out = node(x)
        if np.isnan(out.data.cpu().numpy()[0]):
            break
#        print(out.data.cpu())
        l = loss(out,target)
        l.backward()
        optimizer.step()
        loss_array.append(l.data.cpu().numpy())

    print(f'OUTPUT: {out.data.cpu()}')
    print(list(node.parameters())[0])
    print(list(node.parameters())[-1][:8])
    
    plt.figure()
    plt.plot(loss_array)
    plt.show()
