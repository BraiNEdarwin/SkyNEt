#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 13:23:57 2018

@author: hruiz
"""

import torch
import torch.nn as nn
import numpy as np 

x=torch.randn(109)
x = torch.autograd.Variable(x)
y_pred = x

pred_layer = nn.Linear(1,6)
pred_layer.weight.data[0:5] = torch.zeros_like(pred_layer.weight.data[0:5])

pred_layer.weight.data[5] = torch.ones_like(pred_layer.weight.data[5])
pred_layer.weight.data
pred_layer.weight.requires_grad
pred_layer.weight.requires_grad = False
pred_layer.weight.requires_grad

pred_layer.bias.data
pred_layer.bias.data[-1:]=torch.zeros_like(pred_layer.bias.data[-1:])
pred_layer.bias.data

l_out = nn.Linear(6,1)
l_out.weight.requires_grad

for param in l_out.parameters(): param.requires_grad = False

predictor = nn.Sequential(pred_layer,l_out)
print('####### Parameters before learning ###########')
for params in predictor.parameters(): 

    print(params)
print('####################################') 
      
params_opt = filter(lambda p: p.requires_grad, predictor.parameters())
optim = torch.optim.Adam(params_opt,lr=1)
loss_fn = nn.MSELoss(size_average=True)

for i in range(len(x)):
    y=predictor(x[i])
    loss = loss_fn(y,y_pred[i])
    optim.zero_grad()
    loss.backward()
    for param in predictor.parameters():
        if param.requires_grad:
            param.grad[-1] = 0
          
    optim.step()
    
print('####### Parameters AFTER learning ###########')
for params in predictor.parameters(): 
    print(params)
print('####################################')