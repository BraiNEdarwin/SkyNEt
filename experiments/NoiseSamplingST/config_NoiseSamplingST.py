# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:43:23 2018
Config file corresponding to NoiseSamplingST.py

@author: Mark Boon
"""

import numpy as np


class experiment_config():
    
    
    def __init__(self):
        
        self.fs = 1000          # Sampling freq (Hz)
        self.sampleTime = 2     # Sampling time (s)
        self.amplification = 1E9
        self.steps = [[400, 600],[400, 600],[400, 600],[400],[400],[400],[400]]  # Steps per control voltage
        self.gridHeight = len(self.steps[0])        # Amount of steps for a CV
        self.controls = 7
        self.T_test = True      # Tests variations in the variance for a sample time
        self.S_test = True      # Tests variations in the variance for measure - switch - measure for one CV
        self.samples = 10       # Amount of measurements for one CV config
        
        self.CVsteps = [len(self.steps[i]) for i in range(len(self.steps))]
        self.iterations = np.prod(self.CVsteps)
        
        
        #self.filepath = 'D:\data\Mark\ST_tests'
        self.filepath = 'D:\\data\\Mark\\'
        self.name_T = 'SampleTimeMeas'
        self.name_S = 'SwitchMeas'
        

