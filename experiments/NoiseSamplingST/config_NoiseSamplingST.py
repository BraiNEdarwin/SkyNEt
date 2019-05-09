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
        self.device = 'cDAQ'
        self.experiment_type = 'test_set'

        self.samples = 1       # Amount of measurements for one CV config
        self.sampleTime = 0.1    # Sampling time (s)
        self.name_T = 'test_set4_rand_retry'
        self.name_S = 'SwitchMeas' + str(self.sampleTime) +'s'

        self.fs = 1000          # Sampling freq (Hz)
        self.rampT = int(self.fs/4) # Time to ramp up the input voltages
        self.amplification = 100
        self.postgain = 1
        self.controls = 7
        self.filepath = 'D:\\data\\Mark\\ST_tests\\paper_chip\\'
        # [S2d, matrix module index, electrode on device]
        self.electrodeSetup = [['ao5','ao3','ao1''ao0','a02','ao4','ao6','out'],[1,3,5,6,11,13,15,17],[5,6,7,8,1,2,3,4]]
        # IF CVs are already found, use this:
        #self.CVs = np.array([[671.072,-467.805,-380.494,-341.477,-263.301,887.59,-860.786]])
        #self.CVs = np.array([[0.75,0,0,0,0,0,0],[0.8,0,0,0,0,0,0], [0.85,0,0,0,0,0,0],[0.9,0,0,0,0,0,0],[0.95,0,0,0,0,0,0]]) # Only ao0 is connected, the rest is floating
        self.CVs = np.load(r'D:\data\Mark\wave_search\paper_chip\2019_04_27_115357_train_data_2d_f_0_05\testset4_inputs.npz')['inputs']

    
        #%% Use boolean logic script to find current outputs to use for noise measurement
        self.genelabels = ['CV1/T1','CV2/T3','CV3/T5','CV4/T7','CV5/T11','CV6/T13','CV7/T15','input scaling']
        self.nameCV = 'CVs'
        self.configSrc = os.path.dirname(os.path.abspath(__file__))
        
        ##############################################
        # Additional settings for when the new wave set up is used to sample datapoints
        ##############################################

        self.waveElectrodes = 7
        self.factor = 0.05
        self.freq2 = np.array([2,np.pi,5,7,13,17,19]) 
        self.freq = np.sqrt(self.freq2[:self.waveElectrodes])*self.factor
        self.phase = np.zeros(self.waveElectrodes)
        self.Vmax = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.5, 0.5]) # Maximum amount of voltage for the inputs
        self.offset = np.array([-0.3, -0.3, -0.3, -0.3, -0.3, -0.2, -0.2])
        self.fs_wave = 50 # the fs for this experiment can be different than the fs of the characterization (to see more noise)
        # This array contains all the datapoints to measure (each point will be measure sampleTime seconds):
        self.loadstring = r'D:\data\Mark\wave_search\paper_chip\2019_04_27_115357_train_data_2d_f_0_05\noise_inputs.npz'
        self.t = np.load(self.loadstring)['inputs'][1:,1]
        #self.t = np.array([172180]) # 152799 94573

        self.electrodeSetup = [['ao5','ao3','ao1''ao0','a02','ao4','ao6','out'],[1,3,5,6,11,13,15,17],[5,6,7,8,1,2,3,4]]

        ############# functions ################

        
    def generateSineWave(self, freq, t, amplitude, fs, phase = np.zeros(7)):
        '''
        Generates a sine wave that can be used for the input data.

        freq:       Frequencies of the inputs in an one-dimensional array
        t:          The datapoint(s) index where to generate a sine value (1D array when multiple datapoints are used)
        amplitude:  Amplitude of the sine wave (Vmax in this case)
        fs:         Sample frequency of the device
        phase:      (Optional) phase offset at t=0
        '''

        return np.sin((2 * np.pi * freq[:, np.newaxis] * t )/ fs + phase[:,np.newaxis]) * amplitude[:,np.newaxis] 

        
            
        
        
        
        
        

