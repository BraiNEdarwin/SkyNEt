#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:18:34 2019
Train NN for functionality of dopantNet on the input provided.
---------------
Arguments
inputs : torch.Tensors (shape: NxD) with the correct device specification.
targets   : Target data of shape NxD where N is number of samples, D is output dim. 
filepath (kwarg, str)  : Path to save the results
---------------
Returns:
control_inputs (np.array) : array containing control inputs to the dopantNet
output (np.array)  : array with the output of the dopantNet
cost (np.array)    : array with the costs (train) per epoch

Notes:
    The dopantNet is composed by a surrogate model of a dopant network device
    and bias learnable parameters that serve as control inputs to tune the device for desired functionality.
    The output of the net is evaluated with the same cost function used for the device and
    it is evaluated for separability using a perceptron.

@author: hruiz
"""

