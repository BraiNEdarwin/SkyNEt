# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:43:23 2018
Config file corresponding to NoiseSamplingST.py

@author: Mark Boon
"""


class experiment_config():
    
    
    def __init__(self):
        
        self.fs = 1000          # Sampling freq (Hz)
        self.sampleTime = 2     # Sampling time (s)
        self.amplification = 100E6
        self.steps = [[500, 200],[200],[200],[200],[200],[200],[200]]  # Steps per control voltage
        self.gridHeight = len(self.steps[0])        # Amount of steps for a CV
        self.controls = 7
        self.T_test = True      # Tests variations in the variance for a sample time
        self.S_test = True      # Tests variations in the variance for measure - switch - measure for one CV
        self.samples = 10
        
        
        
        #self.filepath = 'D:\data\Mark\ST_tests'
        self.filepath = 'C:\\Users\\User\APH\\Thesis\Code\\test\\'
        self.name_T = 'SampleTimeData'
        self.name_S = 'SwitchData'
        

