#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:40:12 2018

@author: Mark

This is the light version of the network where you use frequency, amplitude, offset, fs and datapoint index 
information to generate the input values for the minibatches. Therefore you don't need to load the complete input data information
at once in your memory.

INPUT: ->data; a list of (input,output) pairs for training and validation [(x_train,y_train),(x_val,y_val)]. 
               The dimensions of x and y arrays must be (Samples,Dim) and they must be torch.FloatTensor
       ->depth: number of hidden layers
       ->width: can be list of number of nodes in each hidden layer

       ->kwarg: loss='MSE'; string name of loss function that defines the problem (only default implemented)
                activation='ReLU'; name of activation function defining the non-linearity in the network (only default implemented)
                betas=(0.9, 0.999) is a tuple with beta1 and beta2 for Adam
"""
import torch
import numpy as np 
from SkyNEt.modules.Nets.staNNet import staNNet


class lightNNet(staNNet):
    
    def load_data(self, data):
        return self.generateSineWave(self.info['freq'], data, self.info['amplitude'],
                                                     self.info['fs'], self.info['offset'], self.info['phase'])  
        

    def generateSineWave(self,freq, t, amplitude, fs, offset = np.zeros(7), phase = np.zeros(7)):
        '''
        Generates a sine wave that can be used for the input data.

        freq:       Frequencies of the inputs in an one-dimensional array
        t:          The datapoint(s) index where to generate a sine value (1D array when multiple datapoints are used)
        amplitude:  Amplitude of the sine wave (Vmax in this case)
        fs:         Sample frequency of the device
        phase:      (Optional) phase offset at t=0

        '''     
        if torch.cuda.is_available() and isinstance(t,torch.cuda.FloatTensor):
            waves = amplitude * np.sin((2 * np.pi * np.outer(t.cpu(),freq))/ fs + phase) + np.outer(np.ones(t.shape[0]),offset)
            waves = torch.from_numpy(waves).type(torch.cuda.FloatTensor)
        else:
            waves = amplitude * np.sin((2 * np.pi * np.outer(t,freq))/ fs + phase) + np.outer(np.ones(t.shape[0]),offset)
            waves = torch.from_numpy(waves).type(torch.float32)     
            
        return  waves    
