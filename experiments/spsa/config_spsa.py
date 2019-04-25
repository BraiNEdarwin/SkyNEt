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
        self.device = 'NN'
        self.main_dir = r'C:\Users\User\APH\Thesis\Data\wave_search\champ_chip\2019_04_05_172733_characterization_2days_f_0_05_fs_50\nets\MSE_n_proper\\'
        self.NN_name = 'MSE_n_d10w90_300ep_lr3e-3_b1024_b1b2_0.90.75_seed.pt'
        self.inputIndex = [4,5] # Electrodes that will be used as boolean input
        ################################################
        ######### SPECIFY PARAMETERS ###################
        ################################################
        self.comport = 'COM3'  # COM port for the ivvi rack
        #self.filepath = r'D:\Data\Mark\spsa\\'
        self.filepath = r'C:\Users\User\APH\Thesis\Data\SPSA\\'
        
        # Define experiment
        self.postgain = 100
        self.amplification = 1000
        self.gainFactor = self.amplification/self.postgain # gainFactor scales the output such that it is always in order of nA
        self.inputScaling = 0.9
        self.inputOffset = -0.5
        self.CVrange = np.array([[-0.8, 0.2],[-0.8, 0.2],[-1.1, 0.8],[-1.1, 0.8],[-1.1, 0.8]])   # Range for the control voltages
        self.targetGen = self.XNOR
        self.name = "XNOR_NN"
        
        self.inputs = 2
        self.controls = 5
        self.a = 1000 # Initial 'learn rate'
        self.A = .1 # Decay factor of learn rate
        self.c = 50 # Initial stepsize of CVs
        self.alpha = 0.4 # Decay factor of learn rate
        self.gamma = 0.101 # Decay factor of stepsize CVs
        self.n = 100
        
        self.loss = self.cor_separation_loss
        self.CVlabels = ['CV1/T1','CV2/T3','CV3/T11','CV4/T13','CV5/T15', 'Input scaling']
        self.configSrc = os.path.dirname(os.path.abspath(__file__))



    def LossCorr(self, x, target, W):
        '''
        Adapted version such that the return lies between 0 and 100, 100 being the worst fitness.
        '''

        #extract fit data with weights W
        x = x[W.astype(int)==1] # Remove all datapoints where w = 0
        target = target[W.astype(int)==1]

        #x_weighed = np.empty(len(indices))
        #target_weighed = np.empty(len(indices))
        #for i in range(len(indices)):
        #    x_weighed[i] = x[indices[i]]
        #    target_weighed[i] = target[indices[i]]

        F = np.corrcoef(x, target)[0, 1]
        clipcounter = 0
        for i in range(len(x)):
            if(abs(x[i]) > 3.4*10):
                clipcounter = clipcounter + 1
                F = -1
        return 100*abs(1 - (F + 1)/2)
    
    def cor_separation_loss(self, x, t, w):
        x = x[w.astype(int)==1] # Remove all datapoints where w = 0
        t = t[w.astype(int)==1]
        
        #F = np.corrcoef(x, t)[0, 1]
        
        corr = np.mean((x-np.mean(x))*(t-np.mean(t)))/(np.std(x)*np.std(t)+1E-12)
        x_high_min = np.min(x[(t == self.gainFactor)])
        x_low_max = np.max(x[(t == 0)])
        #return 100*(1.0001 - corr)/(abs(x_high_min-x_low_max + 1E-12)/2)**.5
        return (abs(x_high_min-x_low_max + 1E-12)/2)**.5/corr
        #return (1.0001 - F)/(abs(x_high_min-x_low_max + 1E-12)/1)**.5