#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 16:55:07 2018
Here the convolutional neural network for the TURBO project is build
@author: hruiz
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F

class convNet(nn.Module):
    
    #1 input channel, c_out output channels, m-sized kernel
    def __init__(self, L_in, kernel_size = 3, max_pool_size = 2,
                 c1_out = 5, c2_out = 7, hiduni = 50):
        super(convNet,self).__init__()
        
        self.max_pool_size = max_pool_size
        padding = int(kernel_size/2)
        self.conv1 = nn.Conv1d(1, c1_out, kernel_size, padding = padding)
        L1 = (L_in + 2*padding - kernel_size + 1)/self.max_pool_size
        
        kernel_size = 2*kernel_size
        padding = int(kernel_size/2)
        self.conv2 = nn.Conv1d(c1_out, c2_out, kernel_size, padding = padding)
        self.nr_features = c2_out*int((L1 + 2*padding - kernel_size + 1)/self.max_pool_size)
        
        #Affine operations: y = Wx + b
        self.linear1 = nn.Linear(self.nr_features,hiduni)
        self.linear2 = nn.Linear(hiduni,hiduni)
        self.out_layer = nn.Linear(hiduni,1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
#        print(x.shape)
        x = F.max_pool1d(x,self.max_pool_size)
#        print(x.shape)
        x = F.relu(self.conv2(x))
#        print(x.shape)
        x = F.max_pool1d(x,self.max_pool_size)
#        print(x.shape)
        nr_features = self.get_nr_features(x)
        x = x.view(-1, nr_features)
#        print(x.shape)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.out_layer(x)
    
    def get_nr_features(self,x):
        x_shape = x.shape[1:] #Exclude the batch dimension
        nr_features = 1
        for i in x_shape:
            nr_features *= i
#        print('Number of features into all-2-all NN ',nr_features)
        return nr_features

#%% Example        
#L = 337    
#net = convNet(L, kernel_size = 5, max_pool_size =3,
#                 c1_out = 10, c2_out = 10, hiduni = 90)
#print(net)
#params = list(net.parameters())
#samples = 100
#inp = torch.randn(samples, 1, L)
#out = net(inp)
#mse = nn.MSELoss() 
#target = torch.randn((samples,1))
#loss = mse(out,target)
#print(loss)
##print(loss.grad_fn)
#net.zero_grad()
#print('Are there parameters? ',net.conv1.bias.grad)
#loss.backward()
#print('Are there parameters? ',params[0].grad)
#net.zero_grad()
#print('conv1.bias after zeroing \n ',net.conv1.bias.grad)
## Using the optimization package
#import torch.optim as optim
#import numpy as np
#from matplotlib import pyplot as plt
##Create optimizar
#optimizer = optim.SGD(net.parameters(), lr = 0.01)
##Update weights during epochs loops
#epochs = 1000
#training_loss = np.zeros((epochs,1))
#for ep in range(epochs):
#    optimizer.zero_grad()
#    output = net(inp)
#    loss = mse(output,target)
#    loss.backward()
#    optimizer.step()
#    output = net(inp)
#    training_loss[ep] = mse(output,target).data
#    print('Epoch: ', ep)
#    
#plt.figure()
#plt.plot(training_loss)