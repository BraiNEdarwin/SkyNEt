# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 07:55:01 2018

@author: Mark Boon
"""

from SkyNEt.config.config_class import config_class


class experiment_config(config_class):
    
    def __init__(self):
        super().__init__() 
        ################################################
        ######### SPECIFY PARAMETERS ###################
        ################################################
        self.comport = 'COM3'  # COM port for the ivvi rack
        self.filepath = r'D:\Data\Mark\spsa\\'

        # Define experiment
        self.amplification = 10
        self.CVrange = [-900, 900]
        self.targetGen = self.AND
        self.name = "AND"
        
        self.inputs = 2
        self.controls = 5
        self.a = 1 # Initial 'learn rate'
        self.A = .5 # Decay factor of learn rate
        self.c = 50 # Initial stepsize of CVs
        self.alpha = 0.603 # Decay factor of learn rate
        self.gamma = 0.101 # Decay factor of stepsize CVs
        self.n = 100
        
        self.loss = config_class.FitnessNMSE
        self.CVlabels = ['CV1/T1','CV2/T3','CV3/T11','CV4/T13','CV5/T15', 'Input scaling']