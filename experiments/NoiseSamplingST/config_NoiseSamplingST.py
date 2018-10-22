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
        self.sampleTime = 1000    # Sampling time (s)
        self.res = 1E9
        self.steps = [[-623.79],[247.38],[826.70],[-294.96],[-108.88],[668.49],[-563.18]]  # Steps per control voltage
        self.controls = 7
        
        self.findCV = True         # If true: use GA to find CVs for all targetCurrents
        self.gridSearch = False     # If true: use a grid for sampling (use sekf.steps)
        self.T_test = True      # Tests variations in the variance for a sample time
        self.S_test = False      # Tests variations in the variance for measure - switch - measure for one CV
        self.samples = 1       # Amount of measurements for one CV config
        
        self.filepath = 'D:\\data\\Mark\\ST_tests\\'
        self.name_T = 'SampleTimeMeas' + str(self.sampleTime) +'s'
        self.name_S = 'SwitchMeas' + str(self.sampleTime) +'s'
        # [S2d, matrix module index, electrode on device]
        self.electrodeSetup = [[1,2,3,4,5,6,7,'grnd A'],[1,3,5,7,11,13,15,17],[5,6,7,8,1,2,3,4]]
        
        # IF CVs are already found, use this:
        self.CVs = np.array([[285.875,575.273,-539.401,-233.522,-220.229,789.934,-308.457],
[-686.219,763.966,-755.037,-369.096,757.369,790.724,-352.867],
[-493.151,874.259,-256.779,-56.5794,-769.594,-76.1593,-379.087],
[-624.662,-665.318,-156.936,719.789,-563.085,693.766,-736.543],
[419.698,-173.237,520.219,-707.392,-305.152,268.228,-741.344],
[624.015,816.139,-77.78,-187.469,-555.867,403.37,181.379],
[47.7031,-761.01,-469.584,297.609,-579.117,-810.344,699.43],
[545.732,-383.307,-280.978,249.724,354.602,-314.887,394.968],
[76.1954,-749.319,454.49,729.533,-241.704,-807.347,735.137],
[-468.735,-120.546,95.1499,352.884,-320.34,-841.181,678.089],
[10.5064,234.343,638.217,-496.252,-632.652,-825.935,800.611],
[602.51,385.086,226.637,-472.473,-756.897,-864.359,620.725],
[605.027,433.85,21.4779,-702.346,736.333,83.085,-208.909]])
                    
    
    
        #%% Use boolean logic script to find current outputs to use for noise measurement
        
        self.genelabels = ['CV1/T1','CV2/T3','CV3/T5','CV4/T7','CV5/T11','CV6/T13','CV7/T15','input scaling']
        self.nameCV = 'CVs'
        
        self.amplification = 1 
        self.genes = 8              # Must be 8 because boolean_logic defines control voltages for genes - 1
        self.genomes = 25
        self.generations = 10
        self.generange = [[-900,900], [-900, 900], [-900, 900], [-900, 900], [-900, 900], [-900, 900], [-900, 900],[0., 1.]]

        self.targetCurrent = [0.00, 0.25,0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25]    # The desired output current
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
        
        
            
        
        
        
        
        

