# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:43:23 2018
Config file corresponding to NoiseSamplingST.py

@author: Mark Boon
"""

from SkyNEt.config.config_class import config_class
import numpy as np
import os

class experiment_config(config_class):

    
    def __init__(self):
        super().__init__()
        self.device = 'nidaqFull'
        self.findCV = False         # If true: use GA to find CVs for all targetCurrents
        self.gridSearch = False     # If true: use a grid for sampling (use self.steps)
        self.T_test = True      # Tests variations in the variance for a sample time
        self.S_test = False      # Tests variations in the variance for measure - switch - measure for one CV
        self.samples = 1       # Amount of measurements for one CV config
        self.sampleTime = 1000    # Sampling time (s)
        self.name_T = 'characterization_20h_batch_25s_noiseCheck_long' + str(self.sampleTime) +'s'
        self.name_S = 'SwitchMeas' + str(self.sampleTime) +'s'

        self.fs = 5000          # Sampling freq (Hz)
        self.signallength = 2   # Used for GA (s)
        self.edgelength = 0.01  # Used for GA (s)
        self.amplification = 1000
        self.postgain = 100
        self.steps = [[-800,-600,-400,-200,0,200,400,600,800],[-800,-600,-400,-200,0,200,400,600,800],[0],[0],[0],[-800,-600,-400,-200,0,200,400,600,800],[-800,-600,-400,-200,0,200,400,600,800]]  # Steps per control voltage
        self.controls = 7
        self.filepath = 'D:\\data\\Mark\\ST_tests\\'
        # [S2d, matrix module index, electrode on device]
        self.electrodeSetup = [[3,4,5,1,2,6,7,'out'],[1,3,5,7,11,13,15,17],[5,6,7,8,1,2,3,4]]
        # IF CVs are already found, use this:
        self.CVs = np.array([[671.072,-467.805,-380.494,-341.477,-263.301,887.59,-860.786]])
    
        #%% Use boolean logic script to find current outputs to use for noise measurement
        self.genelabels = ['CV1/T1','CV2/T3','CV3/T5','CV4/T7','CV5/T11','CV6/T13','CV7/T15','input scaling']
        self.nameCV = 'CVs'
        self.configSrc = os.path.dirname(os.path.abspath(__file__))
        
        self.genes = 8              # Must be 8 when controlling 7 because boolean_logic defines control voltages for genes - 1
        self.genomes = 25
        self.generations = 5
        self.generange = [[-900,900], [-900, 900], [-900, 900], [-900, 900], [-900, 900], [-900, 900], [-900, 900],[0., 1.]]
        self.targetCurrent = [1.05, 1.1, 1.2]
        #self.targetCurrent = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00]    # The desired output current
        self.TargetGen = self.Target
        self.Fitness = self.FitnessNMSE
        self.fitThres = 2000            #Threshold for high enough fitness value during search
              
        ##############################################
        # Additional settings for when the new wave set up is used to sample datapoints
        ##############################################

        self.waveElectrodes = 7
        self.factor = 2
        self.freq2 = np.array([2,np.pi,5,7,13,17,19]) 
        self.freq = np.sqrt(self.freq2[:self.waveElectrodes])*self.factor
        self.phase = np.zeros(self.waveElectrodes)
        self.Vmax = 0.8 # Maximum amount of voltage for the inputs
        self.fs_wave = 500 # the fs for this experiment can be different than the fs of the characterization (to see more noise)
        # This array contains all the datapoints to measure (each point will be measure sampleTime seconds):
        self.loadstring = r'D:\data\Mark\wave_search\2019_01_30_123621_characterization_20h_batch_25s_fs_500_f_2\noise14TestPoints.npz'
        self.t = np.load(self.loadstring)['sampleData'][1:,1]

        if self.device == 'nidaqFull':
            self.electrodeSetup = [['ao0','ao2','ao4''ao6','a05','ao3','ao1','out'],[1,3,5,7,11,13,15,17],[5,6,7,8,1,2,3,4]]

        ############# functions ################


    def Target(self):        # Dummy function so that the boolean_logic script can be used
        samples = 4 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)
        t = np.linspace(0, samples/self.fs, samples)
        x = np.zeros((len(self.targetCurrent), samples))
        for i in range(len(self.targetCurrent)):
            x[i,:] = self.targetCurrent[i] * np.ones((samples))
        return t, x
        
    def generateSineWave(self, freq, t, amplitude, fs, phase = np.zeros(7)):
        '''
        Generates a sine wave that can be used for the input data.

        freq:       Frequencies of the inputs in an one-dimensional array
        t:          The datapoint(s) index where to generate a sine value (1D array when multiple datapoints are used)
        amplitude:  Amplitude of the sine wave (Vmax in this case)
        fs:         Sample frequency of the device
        phase:      (Optional) phase offset at t=0
        '''

        return np.sin((2 * np.pi * freq[:, np.newaxis] * t + phase[:,np.newaxis])/ fs) * amplitude
        
            
        
        
        
        
        

