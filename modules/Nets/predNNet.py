#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 14:39:57 2018

@author: hruiz
"""

import torch
import torch.nn as nn
import numpy as np 
from Nets.staNNet import staNNet

class predNNet(staNNet):
    
    def __init__(self,*args,loss='MSE',C=1.,activation='ReLU',BN=False):
        super(predNNet,self).__init__(*args,loss=loss,C=C,activation=activation,BN=BN)
    
    def _construct_predictor(self):
        model = self.model
        for param in model.parameters():
            param.requires_grad = False
        pred_layer = nn.Linear(self.pred_in,self.D_in)
        self.dim_cv = self.D_in - self.pred_in
        pred_layer.bias.data.uniform_(0.01,0.5)
#        print('Predictor Bias initialized with uniform: ',pred_layer.bias.data)
        #set weights to zero to supress mixture of input and control voltages
        pred_layer.weight.data[-self.dim_cv:] = torch.zeros_like(
                pred_layer.weight.data[-self.dim_cv:])
        pred_layer.weight.data[:self.pred_in] = torch.eye(2)
        pred_layer.weight.requires_grad = False
        #set bias terms for inputs to zero
        pred_layer.bias.data[:self.pred_in] = torch.zeros_like(
                pred_layer.bias.data[:self.pred_in]) 
#        print('Predictor Bias set to zero for inputs: ',pred_layer.bias.data)
        
        self.predictor = nn.Sequential(pred_layer,model)
        
    def cv_regularizer(self,strength=0.00001,boundary=0.001):
        reg_loss =0 
        for name, param in self.predictor.named_parameters(): 
            if name == '0.bias': 
                x = param[self.pred_in:]
                reg_loss = (x-0.5)**6 + strength*(1/(x+boundary)**2+1/(x-(0.8-boundary))**2)
#                print('param = ',param)
#                reg_loss = (2*abs(x-0.5))**50
                reg_loss = torch.sum(reg_loss)
#                print('reg_loss = ',reg_loss)
        return reg_loss
    
    def predict(self,data,learning_rate,nr_epochs,batch_size,betas=(0.9,0.999),
                scale=1.0,reg_scale=0.0,seed=False):
        
        scale = torch.FloatTensor([scale])
        print('Scale is ',scale)
        x_train, y_train = data[0]
        yt_var = torch.FloatTensor([np.var(y_train.data.numpy())])
        
        x_val, y_val = data[1]
        self.pred_in = x_train.shape[1]
        if seed:
            torch.manual_seed(22)
            print('The torch RNG is seeded!')
            
        self._construct_predictor()
        params_opt = filter(lambda p: p.requires_grad, self.predictor.parameters())
#        optimizer = torch.optim.SGD(params_opt, lr=learning_rate,momentum=0.7,nesterov=True) 
        optimizer = torch.optim.Adam(params_opt, lr=learning_rate,betas=betas) # OR SGD?!
        print('Prediction using ADAM optimizer')
        self.grads_epoch = np.zeros((nr_epochs,self.dim_cv))
        self.cv_epoch = np.zeros((nr_epochs,self.dim_cv))
        valErr_pred = np.zeros((nr_epochs,))
        for epoch in range(nr_epochs):
            
            permutation = torch.randperm(len(x_train))#.type(self.itype) # Permute indices 
            running_loss = 0.0 
            nr_minibatches = 0
            grads = 0.
            
            for i in range(0,len(permutation),batch_size):            

                # Forward pass: compute predicted y by passing x to the model.
                indices = permutation[i:i+batch_size]
                y_pred = self.predictor(x_train[indices])
                
                # Compute and print loss.
#                loss = self.loss_fn(y_pred, y_train[indices])
                loss = self.loss_fn(y_pred, y_train[indices]) + reg_scale*self.cv_regularizer()
                loss = loss*scale/yt_var
                
                running_loss += loss.item()      
                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable
                # weights of the model). This is because by default, gradients are
                # accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called. Checkout docs of torch.autograd.backward for more details.
                optimizer.zero_grad()
                
                # Backward pass: compute gradient of the loss with respect to model
                # parameters
                loss.backward()
                # Zero the gradients of the inpout-bias
                for param in self.predictor.parameters():
                    if param.requires_grad: 
                        param.grad[:self.pred_in] = 0
#                        print('pre update: ', param.grad)
                        grads += param.grad[self.pred_in:]
                # Calling the step function on an Optimizer makes an update to its
                # parameters
                optimizer.step()
                nr_minibatches += 1
            
            self.grads_epoch[epoch] = grads/nr_minibatches
            for param in self.predictor.parameters():
                if param.requires_grad: self.cv_epoch[epoch] = param[self.pred_in:].data.numpy()
                
            y_pred = self.predictor(x_val)
#            loss = self.loss_fn(y_pred, y_val)
            loss = self.loss_fn(y_pred, y_val) + reg_scale*self.cv_regularizer()
            loss = loss*scale/yt_var
            
            valErr_pred[epoch] = loss.item()

        print('Val. Error @ END:', loss.item(),'Training Error:',running_loss/nr_minibatches)
        for param in self.predictor.parameters():
            if param.requires_grad: control_voltages = param[self.pred_in:]
        return control_voltages.cpu().detach().numpy() , valErr_pred