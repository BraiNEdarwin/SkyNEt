# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 16:08:49 2019

@author: hansr
"""
import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class single_layer_net(nn.Module):
    
    def __init__(self,nr_hidden_neurons):
        super().__init__()
        
        self.in_layer = nn.Linear(2,nr_hidden_neurons).to(device)
        self.hidden_layer = nn.Linear(nr_hidden_neurons,1).to(device)
        self.activation = nn.Sigmoid()
        
    def forward(self,x):
        
        x = self.in_layer(x)
        x = self.activation(x)
        x = self.hidden_layer(x)
        
        return self.activation(x)