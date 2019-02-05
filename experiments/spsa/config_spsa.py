# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 07:55:01 2018

@author: Mark Boon
"""

from SkyNEt.config.config_class import config_class
import numpy as np
import os


class experiment_config(config_class):
    
    def __init__(self):
        super().__init__() 
        ################################################
        ######### SPECIFY PARAMETERS ###################
        ################################################
        self.comport = 'COM3'  # COM port for the ivvi rack
        self.filepath = r'D:\Data\Mark\spsa\\'

        # Define experiment
        self.postgain = 100
        self.amplification = 1000
        self.gainFactor = self.amplification/self.postgain # gainFactor scales the output such that it is always in order of nA
        self.CVrange = [-800, 800]
        self.targetGen = self.AND
        self.name = "AND"
        
        self.inputs = 2
        self.controls = 5
        self.a = 10000 # Initial 'learn rate'
        self.A = .1 # Decay factor of learn rate
        self.c = 100 # Initial stepsize of CVs
        self.alpha = 0.4 # Decay factor of learn rate
        self.gamma = 0.101 # Decay factor of stepsize CVs
        self.n = 100
        
        self.loss = self.LossCorr   
        self.CVlabels = ['CV1/T1','CV2/T3','CV3/T11','CV4/T13','CV5/T15', 'Input scaling']
        self.configSrc = os.path.dirname(os.path.abspath(__file__))



    def LossCorr(self, x, target, W):
        '''
        Adapted version such that the return lies between 0 and 100, 100 being the worst fitness.
        '''

        #extract fit data with weights W
        indices = np.argwhere(W)  #indices where W is nonzero (i.e. 1)

        #x_weighed = np.empty(len(indices))
        #target_weighed = np.empty(len(indices))
        #for i in range(len(indices)):
        #    x_weighed[i] = x[indices[i]]
        #    target_weighed[i] = target[indices[i]]

        F = np.corrcoef(x, target)[0, 1]
        clipcounter = 0
        for i in range(len(x)):
            if(abs(x[i]) > 3.1*10):
                clipcounter = clipcounter + 1
                F = -1
        return 100*abs(1 - (F + 1)/2)