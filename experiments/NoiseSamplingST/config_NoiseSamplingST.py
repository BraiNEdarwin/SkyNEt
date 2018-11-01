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
        self.signallength = 2   # Used for GA (s)
        self.edgelength = 0.01  # Used for GA (s)
        self.sampleTime = 5000    # Sampling time (s)
        self.res = 1E9
        self.steps = [[-800,-600,-400,-200,0,200,400,600,800],[-800,-600,-400,-200,0,200,400,600,800],[0],[0],[0],[-800,-600,-400,-200,0,200,400,600,800],[-800,-600,-400,-200,0,200,400,600,800]]  # Steps per control voltage
        self.controls = 7
        
        self.findCV = False         # If true: use GA to find CVs for all targetCurrents
        self.gridSearch = False     # If true: use a grid for sampling (use self.steps)
        self.T_test = True      # Tests variations in the variance for a sample time
        self.S_test = False      # Tests variations in the variance for measure - switch - measure for one CV
        self.samples = 1       # Amount of measurements for one CV config
        
        self.filepath = 'D:\\data\\Mark\\ST_tests\\'
        self.name_T = 'SampleTimeMeas' + str(self.sampleTime) +'s'
        self.name_S = 'SwitchMeas' + str(self.sampleTime) +'s'
        # [S2d, matrix module index, electrode on device]
        self.electrodeSetup = [[1,2,3,4,5,6,7,'out'],[1,3,5,7,11,13,15,17],[5,6,7,8,1,2,3,4]]
        
        # IF CVs are already found, use this:
        self.CVs = np.array([[-271.975,-760.161,562.445,518.329,145.908,705.555,-806.885],
[564.963,146.272,576.712,-342.936,139.447,-532.064,508.789],
[4.07539,495.843,-266.899,-324.84,6.39452,814.245,-318.706],
[41.1826,-242.402,-721.716,308.773,-188.442,775.443,-362.874],
[-262.662,-157.5,542.459,235.443,464.925,850.41,884.502],
[488.295,-771.281,549.649,-337.996,339.306,66.1638,-54.8996],
[-404.857,-279.603,-580.696,-428.617,-201.309,-698.45,608.057],
[-72.2634,-854.05,606.87,-684.301,518.955,216.692,780.177],
[-351.778,-799.656,646.86,-203.034,-276.96,352.817,758.577],
[-785.115,-228.673,-506.417,-878.549,-470.357,-671.32,603.476],
[662.221,553.607,441.048,-681.078,-388.27,-15.6731,644.703],
[434.579,-778.963,-85.6676,231.625,-3.35899,-848.032,802.783]])
                    
    
    
        #%% Use boolean logic script to find current outputs to use for noise measurement
        
        self.genelabels = ['CV1/T1','CV2/T3','CV3/T5','CV4/T7','CV5/T11','CV6/T13','CV7/T15','input scaling']
        self.nameCV = 'CVs'
        
        self.amplification = 1 
        self.genes = 8              # Must be 8 when controlling 7 because boolean_logic defines control voltages for genes - 1
        self.genomes = 25
        self.generations = 10
        self.generange = [[-900,900], [-900, 900], [0, 0], [0, 0], [0, 0], [-900, 900], [-300, -900],[0., 1.]]

        #self.targetCurrent = [-1.50, -1.75, -2.00, -2.25, -2.50, -2.75]
        #self.targetCurrent = [-3.00, -2.75, -2.50, -2.25, -2.00, -1.75, -1.50, -1.25, -1.00, -0.75, -0.50, -0.25, 
        self.targetCurrent = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00]    # The desired output current
        #self.targetCurrent = [3.25, 3.00, 2.75, 2.50, 2.25, 2.00, 1.75, 1.50, 1.25, 1.00, 0.75, 0.50, 0.25, 0.00]
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
        
        
            
        
        
        
        
        

