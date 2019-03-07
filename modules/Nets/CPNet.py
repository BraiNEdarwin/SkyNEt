#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 14:39:57 2018

@author: hruiz
"""

import torch
import torch.nn as nn
import numpy as np 
from SkyNEt.modules.Nets.staNNet import staNNet

class CPNet(staNNet):
    
    def __init__(self,*args,loss='MSE',activation='ReLU',BN=False):
        
        super(CPNet,self).__init__(*args,loss=loss,activation=activation,BN=BN)
    
    def _construct_predictor(self):
        model = self.model
        for param in model.parameters():
            param.requires_grad = False
        pred_layer = nn.Linear(self.pred_in,self.D_in)
        self.dim_cv = self.D_in - self.pred_in
        pred_layer.bias.data.uniform_(0.0,1.0)
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
        score_layer = nn.Linear(1,3)
        score_layer.weight.data[1] = 0.
        score_layer.bias.data[1] = 0.
        self.predictor = nn.Sequential(pred_layer,model,score_layer)
        for name, params in self.predictor.named_parameters():
            print('Name: ',name, '| shape:',params.shape,'| Requires grad:',params.requires_grad)
            if name in ['2.weight','2.bias']: print(params)
            
    def cv_regularizer(self):
        reg_loss =0 
        for name, param in self.predictor.named_parameters(): 
            if name == '0.bias': 
                x = param[self.pred_in:]
                reg_loss = torch.relu(-x) + torch.relu(x-1.0)
                reg_loss = torch.sum(reg_loss)
#                print('reg_loss = ',reg_loss)
        return reg_loss
    
    def score_reg(self):
        scr_loss = 0
        for name, param in self.predictor.named_parameters(): 
            if name in ['2.weight','2.bias']:
                scr_loss += torch.norm(param)**2
        return self.lambda_scr*scr_loss
    
    def predict(self, data, learning_rate, nr_epochs, batch_size,
                betas=(0.9,0.999), reg_scale=2.0, lambda_scr=0.0, seed=False):
        self.lambda_scr=lambda_scr
        x_train, y_train = data[0]
#        yt_var = torch.var(y_train)
        
        x_val, y_val = data[1]
        self.pred_in = x_train.shape[1]
        if seed:
            torch.manual_seed(2345)
            print('The torch RNG is seeded!')
            np.random.seed(4367)
            
        self._construct_predictor()
        params_opt = filter(lambda p: p.requires_grad, self.predictor.parameters())
#        optimizer = torch.optim.SGD(params_opt, lr=learning_rate,momentum=0.7,nesterov=True) 
#        print('params_opt: ',list(params_opt))
        optimizer = torch.optim.Adam(params_opt, lr=learning_rate,betas=betas) # OR SGD?!
        print('Prediction using ADAM optimizer')
        self.score_params = np.zeros((2,nr_epochs,3))
        self.cv_epoch = np.zeros((nr_epochs,self.dim_cv))
        valErr_pred = np.zeros((nr_epochs,))
        for epoch in range(nr_epochs):
            
            permutation = torch.randperm(len(x_train)) # Permute indices 
            running_loss = 0.0 
            nr_minibatches = 0
            
            for i in range(0,len(permutation),batch_size):            

                # Forward pass: compute predicted y by passing x to the model.
                indices = permutation[i:i+batch_size]
#                print('predictor is cuda? ', next(self.predictor.parameters()).is_cuda)
#                print('type of x_train ',x_train.type())
#                print('type of indices ',indices.type())
                y_pred = self.predictor(x_train[indices])
#                print('Shape of net output: ',y_pred.shape)
#                print('Shape of y_train[indices]: ',y_train[indices,0].shape)
                # Compute and print loss
                loss = self.loss_fn(y_pred, y_train[indices,0]) + reg_scale*self.cv_regularizer()
                loss = loss + self.score_reg()
                
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
                # Zero the gradients of the input-bias
#                print('Predictor parameters: ',list(self.predictor.parameters()))
                for name, param in self.predictor.named_parameters():
                    if param.requires_grad and name=='0.bias': 
                        param.grad[:self.pred_in] = 0.
                    if param.requires_grad and name in ['2.weight','2.bias']:
                        param.grad[1] = 0.
#                        print('pre update: ', param.grad)
                # Calling the step function on an Optimizer makes an update to its
                # parameters
                optimizer.step()
                nr_minibatches += 1
            
            for name,param in self.predictor.named_parameters():
                if param.requires_grad:
                    if name=='0.bias':
                        self.cv_epoch[epoch] = param[self.pred_in:].data.numpy()
                    elif name == '2.weight':
#                        print(param.data.numpy().shape)
                        self.score_params[0,epoch] = param.data.numpy()[:,0]
                    elif name == '2.bias':
                        self.score_params[1,epoch] = param.data.numpy()
                
            y_pred = self.predictor(x_val)
            loss = self.loss_fn(y_pred, y_val[:,0])
            valErr_pred[epoch] = loss.item()
            print('Epoch',epoch,'Training Error: ', running_loss/batch_size)

        print('Val. Error @ END:', loss.item())
        for name, param in self.predictor.named_parameters():
            if param.requires_grad:
                if name == '0.bias': 
                    control_voltages = param[self.pred_in:].cpu().detach().numpy()
                elif name == '2.weight':
                    self.score_weights= param.cpu().detach().numpy()[:][:,0]
                elif name == '2.bias':
                    self.score_bias= param.cpu().detach().numpy()
        
        return control_voltages, valErr_pred