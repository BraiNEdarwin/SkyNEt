import numpy as np
from SkyNEt.config.config_class import config_class
import os

class experiment_config(config_class):

    def __init__(self):
        super().__init__() 

        self.waveElectrodes = 7
        self.chargeTime = 500 # Static charge time in seconds    
        self.amplitudes = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1., 1.05, 1.1, 1.15, 1.2, 1.2, \
        1.15, 1.1, 1.05, 1., 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.3, 0.2, 0.1]*8) # Magnitude of the voltages for charging

        self.factor = 0.1
        self.freq2 = np.array([2,np.pi,5,7,13,17,19]) 
        self.freq = np.sqrt(self.freq2[:self.waveElectrodes])*self.factor        
        self.phase = np.zeros(self.waveElectrodes)        
        self.fs = 500
        self.t_static = 21230//2 * np.ones(self.fs*10) # static measurement of one input config (made by filling in t_static into generateSineWave)
        self.refAmplitude = 0.3
        self.amplification = 100        
        self.gain_info = '10MV/A'
        self.postgain = 1

        self.samplePoints = int(50*self.fs)
        #                               Summing module S2d      Matrix module           device                
        self.electrodeSetup = [['ao5','ao3','ao1','ao0','a02','ao4','ao6','out'],[1,3,5,6,11,13,15,17],[5,6,7,8,1,2,3,4]]
        # Save settings        
        self.filepath = r'D:\data\Mark\charging_test\paper_chip\\'        
        self.name = 'name'

        self.configSrc = os.path.dirname(os.path.abspath(__file__))    


    def generateSineWave(self, freq, t, amplitude, fs, phase = np.zeros(7)):
        '''
        Generates a sine wave that can be used for the input data.
        freq:       Frequencies of the inputs in an one-dimensional array
        t:          The datapoint(s) index where to generate a sine value (1D array when multiple datapoints are used)
        amplitude:  Amplitude of the sine wave (Vmax in this case)
        fs:         Sample frequency of the device
        phase:      (Optional) phase offset at t=0
        '''

        return np.sin((2 * np.pi * freq[:, np.newaxis] * t)/ fs + phase[:,np.newaxis]) * amplitude[:,np.newaxis]
