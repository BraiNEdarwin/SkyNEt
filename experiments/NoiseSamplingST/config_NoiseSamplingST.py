# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:43:23 2018
Config file corresponding to NoiseSamplingST.py

@author: Mark Boon
"""

from config.config_class import config_class
import numpy as np


class experiment_config(config_class):
    
    
    def __init__(self):
        super().__init__()
        self.fs = 1000          # Sampling freq (Hz)
        self.sampleTime = 2     # Sampling time (s)
        self.res = 1E9
        #self.steps = [[400, 600, 700, 800, 900],[200],[200],[200],[200],[200],[200]]  # Steps per control voltage
        #self.gridHeight = len(self.steps[0])        # Amount of steps for a CV
        #self.controls = 7
        
        self.T_test = True      # Tests variations in the variance for a sample time
        self.S_test = False      # Tests variations in the variance for measure - switch - measure for one CV
        self.samples = 10       # Amount of measurements for one CV config
        
        # self.CVsteps = [len(self.steps[i]) for i in range(len(self.steps))]
        # self.iterations = np.prod(self.CVsteps)
        
        self.filepath = 'D:\\data\\Mark\\ST_tests\\'
        self.name_T = 'SampleTimeMeas'
        self.name_S = 'SwitchMeas'
        
        # [S2d, matrix module index, electrode on device]
        self.electrodeSetup = [[1,2,3,4,5,6,7,'grnd A'],[1,3,5,7,11,13,15,17],[5,6,7,8,1,2,3,4]]
        
        #%% Use boolean logic script to find current outputs to use for noise measurement
        
        self.genelabels = ['CV1/T1','CV2/T3','CV3/T5','CV4/T7','CV5/T11','CV6/T13','CV7/T15','input scaling']
        self.name = 'CVs'
        self.targetCurrent = 0.5    # The desired output current
        self.amplification = 1 
        self.genes = 8              # Must be 8 because boolean_logic defines control voltages for genes - 1
        self.genomes = 20
        self.generations = 10
        self.generange = [[-900,900], [-900, 900], [-900, 900], [-900, 900], [-900, 900], [-900, 900], [-900, 900]]
        self.TargetGen = self.Target
        self.Fitness = self.FitnessNMSE
        
        ################################################
        ################# OFF-LIMITS ###################
        ################################################
        # Check if genomes parameter has been changed
        if(self.genomes != sum(self.default_partition)):
            if(self.genomes%5 == 0):
                self.partition = [self.genomes%5]*5  # Construct equally partitioned genomes
            else:
                print('WARNING: The specified number of genomes is not divisible by 5.'
                      + ' The remaining genomes are generated randomly each generation. '
                      + ' Specify partition in the config instead of genomes if you do not want this.')
                self.partition = [self.genomes//5]*5  # Construct equally partitioned genomes
                self.partition[-1] += self.genomes%5  # Add remainder to last entry of partition

        self.genomes = sum(self.partition)  # Make sure genomes parameter is correct
        self.genes = len(self.generange)  # Make sure genes parameter is correct
        
        
    def Target(self):        # Dummy function so that the boolean_logic script can be used
        samples = 4 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)
        t = np.linspace(0, samples/self.fs, samples)
        x = self.targetCurrent * np.ones((samples))
        return t, x
        
        
            
        
        
        
        
        

