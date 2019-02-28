import numpy as np
from SkyNEt.config.config_class import config_class
import os

class experiment_config(config_class):
    '''
    This is the config for the gradient descent experiment
    
    '''

    def __init__(self):
        super().__init__() #DO NOT REMOVE!
        #######################
        # Physical parameters #
        #######################
        self.controls = 5
        self.freq = np.array([3, 5, 7, 11, 13])  #
        self.fs = 5000
        self.n = 20                 # Amount of iterations
        self.amplification = 1000
        self.postgain = 100
        self.Vrange = [-0.9, 0.9]   # Maximum voltage for the inputs
        self.waveAmplitude = 0.1    # Amplitude of the waves used in the controls
        self.rampT = 0.05           # time to ramp up and ramp down the voltages at start and end of a measurement.
        #                        Summing module S2d      Matrix module           device
        # For the first array: 7 is always the output, 0 corresponds to ao0, 1 to ao1 etc.
        self.electrodeSetup = [[0,1,2,3,4,5,6,7],[1,3,5,7,11,13,15,17],[5,6,7,8,1,2,3,4]]
        
        self.controlLabels = ['ao0','ao1','ao2','ao3','ao4','ao5']
        self.inputIndex = [2,4] # Electrodes that will be used as boolean input
        
        ###################
        # rest parameters #
        ###################
        # parameters for methods
        self.signallength = 1.5  #in seconds
        self.edgelength = 0.01  #in seconds
        
        self.fft_N = 1000       # To obtain an accurate FFT, select freq and N such that 
        self.phase_thres = 0.3
        self.eta = 1            # Learn rate 
        self.errorFunct = self.MSEloss
        self.keithley_address = 'GPIB0::17::INSTR'
        # Save settings
        self.filepath = r'D:\\data\\Mark\\gradient_descent\\'
        self.name = 'gradient_descent'
        self.configSrc = os.path.dirname(os.path.abspath(__file__))
        self.gainFactor = self.amplification/self.postgain


    def MSEloss(self, y, targets, gradients, weights):
        ''' Calculates the mean squared error loss given the gradient of the 
        output w.r.t. the input voltages. This function calculates the error
        for each control separately '''        
        return abs(np.sum((y - targets) * weights)) / np.sum(weights) * gradients
        