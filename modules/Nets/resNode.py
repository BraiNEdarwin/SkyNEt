# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:13:26 2019

@author: Jardi
"""

import torch
import numpy as np 
from SkyNEt.modules.Nets.staNNet import staNNet

class resNode(staNNet):
    
    def __init__(self, *args, loss = 'MSE', activation = 'ReLU', dim_cv = 5, BN = False):
            super(resNode,self).__init__(*args, loss = loss, activation = activation,dim_cv = dim_cv, BN = BN)
            
            self.vlow1 = -0.8
            self.vhigh1 = 0.2
            self.vlow2 = -1.1
            self.vhigh2 = 0.7
           
    def init_cvs(self, input_electrode = 6):
        self.input_electrode = input_electrode
        
        #construct random control voltages
        self.voltages = np.random.uniform(self.vlow1, self.vhigh1, 2)
        self.voltages = np.append(self.voltages, np.random.uniform(self.vlow2, self.vhigh2, 5))
            
    def add_feedback(self, feedback_electrode, feedback_delay = 5, init_value = 0, clow = -30, chigh = 30, w = 1):
        self.feedback_electrode = feedback_electrode
        if init_value is not None:
            self.feedback_init = init_value
            self.voltages[feedback_electrode] = init_value
        else:
            self.feedback_init = self.voltages[feedback_electrode]
        self.feedback_delay = feedback_delay
        if feedback_electrode == 0 or feedback_electrode == 1:
            self.vhigh, self.vlow = self.vhigh1, self.vlow1
        else:
            self.vhigh, self.vlow = self.vhigh2, self.vlow2
        self.feedback_resistance = w * (self.vhigh - self.vlow)/(chigh - clow)
        
    def _feedback_input(self, output, vlow = -1.1, vhigh = 0.7):
        out = self.feedback_init + self.feedback_resistance * output
        if out < vlow:
            out = vlow
        elif out > vhigh:
            out = vhigh
        return out
    
    def _construct_inputs(self, inpt):
        inputs = self.voltages.reshape(1, 7)
        inputs = np.tile(inputs, len(inpt))
        inputs = torch.from_numpy(inputs).float().view(-1, 7)
        inputs[:, self.input_electrode] = torch.from_numpy(inpt).float()
        return inputs

    def get_MC(self, inpt, output):
        N = len(output)
        MC = np.zeros_like(output[:int(N/5)])
        for k, _ in enumerate(output[1:int(N/5)+1], 1):
            MC[k - 1] = np.corrcoef((inpt[:-k], output[k:]))[0,1]**2
        return MC
    
    def outputs(self,inpt):
        inputs = self._construct_inputs(inpt)
        if hasattr(self, 'feedback_electrode'):
            output = np.full_like(inputs.numpy()[:,0], np.nan, dtype=np.double)
            output[0:self.feedback_delay] = self.model(inputs[0:self.feedback_delay,:]).data.cpu().numpy()[:,0]
            for n, _ in enumerate(inputs[self.feedback_delay:], self.feedback_delay):
                inputs[n, self.feedback_electrode] = self._feedback_input(output[n-self.feedback_delay], self.vlow, self.vhigh)
                output[n] = self.model(inputs[n,:]).data.cpu().numpy()
            return output, inputs.numpy()
        else:
            return self.model(inputs).data.cpu().numpy()[:,0], inputs.numpy()