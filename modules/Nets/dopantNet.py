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
        bias = self.min_voltage[self.indx_cv] + \
            (self.max_voltage[self.indx_cv] - self.min_voltage[self.indx_cv])* \
            np.random.rand(1,self.nr_cv)
            
        bias = torch.tensor(bias,dtype=torch.float32).to(device)
        self.bias = nn.Parameter(bias)
        #Set as torch Tensors and send to device
        self.indx_cv = torch.tensor(self.indx_cv,dtype=torch.int64).to(device) #IndexError: tensors used as indices must be long, byte or bool tensors
        self.amplification = torch.tensor(self.net.info['amplification']).to(device)
        self.min_voltage = torch.tensor(self.min_voltage,dtype=torch.float32).to(device)
        self.max_voltage  = torch.tensor(self.max_voltage,dtype=torch.float32).to(device)
    
    def forward(self,x):
        
        expand_cv = self.bias.expand(x.size()[0],-1)
        inp = torch.empty((x.size()[0],x.size()[1]+self.nr_cv)).to(device)
#        print(x.dtype,self.amplification.dtype)

        inp[:,self.in_list] = x
        
#        print(inp.dtype,self.indx_cv.dtype,expand_cv.dtype)
        inp[:,self.indx_cv] = expand_cv
        
        return self.net.model(inp)*self.amplification
    
    
    def regularizer(self):
        x = self.bias
        low = self.min_voltage[self.indx_cv]
        high = self.max_voltage[self.indx_cv]
#        print(x.dtype,low.dtype,high.dtype)
        assert any(low<0), \
        "Min. Voltage is assumed to be negative, but value is positive!"
        assert any(high>0), \
        "Max. Voltage is assumed to be positive, but value is negative!"
        reg = torch.sum( torch.relu(low - x) + torch.relu(x - high) ) 
        return reg
        


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    x = 0.5*np.random.randn(10,3) 
    x = torch.Tensor(x).to(device)
    
    target = torch.Tensor(np.random.randn(10,1)).to(device)
    
    node = dopantNet([0,3,5])
    loss = nn.MSELoss()

#    optimizer = torch.optim.SGD([{'params':filter(lambda p: p.requires_grad, node.parameters())}],lr=0.0001)
    optimizer = torch.optim.SGD([{'params':node.parameters()}],lr=0.0001)
    
    loss_array = []
    change_params_net = []
    change_params0 = []
    
    start_params = [p.clone().detach() for p in node.parameters()]
    
    for i in range(2000):
        
        optimizer.zero_grad()         
        out = node(x)
        if np.isnan(out.data.cpu().numpy()[0]):
            break
#        print(out.data.cpu())
        l = loss(out,target) + node.regularizer()
        l.backward()
        optimizer.step()
        loss_array.append(l.data.cpu().numpy())
        print(loss_array[-1])
        current_params = [p.clone().detach() for p in node.parameters()]
        diff_params = [(current-start).sum() for current,start in zip(current_params,start_params)]
        change_params0.append(diff_params[0])
#        print(diff_params[0].detach())
        change_params_net.append(sum(diff_params[1:]))

    end_params = [p.clone().detach() for p in node.parameters()]
    print("CV params at the beginning: \n ",start_params[0])
    print("CV params at the end: \n",end_params[0])
    print("Example params at the beginning: \n",start_params[-1][:8])
    print("Example params at the end: \n",end_params[-1][:8])
    print("Length of elements in node.parameters(): \n",[len(p) for p in end_params])
    print("and their shape: \n",[p.shape for p in end_params])# if len(p)==1])
    print(f'OUTPUT: {out.data.cpu()}')
    
    plt.figure()
    plt.plot(loss_array)
    plt.title("Loss per epoch")
    plt.show()
    plt.figure()
    plt.plot(change_params0)
    plt.plot(change_params_net)
    plt.title("Difference of parameter with initial params")
    plt.legend(["CV params","Net params"])
    plt.show()
