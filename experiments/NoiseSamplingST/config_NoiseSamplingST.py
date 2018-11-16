# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:43:23 2018
Config file corresponding to NoiseSamplingST.py

@author: Mark Boon
"""

from SkyNEt.config.config_class import config_class
import numpy as np


class experiment_config(config_class):

    
    def __init__(self):
        super().__init__()
        self.fs = 1000          # Sampling freq (Hz)
        self.signallength = 2   # Used for GA (s)
        self.edgelength = 0.01  # Used for GA (s)
        self.sampleTime = 100    # Sampling time (s)
        self.res = 1E9
        self.amplification = 1
        self.steps = [[-800,-600,-400,-200,0,200,400,600,800],[-800,-600,-400,-200,0,200,400,600,800],[0],[0],[0],[-800,-600,-400,-200,0,200,400,600,800],[-800,-600,-400,-200,0,200,400,600,800]]  # Steps per control voltage
        self.controls = 7
        
        self.findCV = False         # If true: use GA to find CVs for all targetCurrents
        self.gridSearch = False     # If true: use a grid for sampling (use self.steps)
        self.T_test = True      # Tests variations in the variance for a sample time
        self.S_test = False      # Tests variations in the variance for measure - switch - measure for one CV
        self.samples = 1       # Amount of measurements for one CV config
        
        self.filepath = 'D:\\data\\Mark\\bandwidth\\'
        self.name_T = 'Tmeas' + str(self.sampleTime) +'s'
        self.name_S = 'SwitchMeas' + str(self.sampleTime) +'s'
        # [S2d, matrix module index, electrode on device]
        self.electrodeSetup = [[3,4,5,1,2,6,7,'out'],[1,3,5,7,11,13,15,17],[5,6,7,8,1,2,3,4]]
        
        # IF CVs are already found, use this:
        self.CVs = np.array([[-280.978,249.724,545.732,-383.307,354.602,-314.887,394.968]])
    
        #%% Use boolean logic script to find current outputs to use for noise measurement
        self.genelabels = ['CV1/T1','CV2/T3','CV3/T5','CV4/T7','CV5/T11','CV6/T13','CV7/T15','input scaling']
        self.nameCV = 'CVs'
        
        self.genes = 8              # Must be 8 when controlling 7 because boolean_logic defines control voltages for genes - 1
        self.genomes = 25
        self.generations = 10
        self.generange = [[-900,900], [-900, 900], [0, 0], [0, 0], [0, 0], [-900, 900], [-300, -900],[0., 1.]]
        self.targetCurrent = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00]    # The desired output current
        self.TargetGen = self.Target
        self.Fitness = self.FitnessNMSE
        self.fitThres = 1000            #Threshold for high enough fitness value during search
              
        
    def Target(self):        # Dummy function so that the boolean_logic script can be used
        samples = 4 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)
        t = np.linspace(0, samples/self.fs, samples)
        x = np.zeros((len(self.targetCurrent), samples))
        for i in range(len(self.targetCurrent)):
            x[i,:] = self.targetCurrent[i] * np.ones((samples))
        return t, x
        
        
            
        
        
        
        
        

