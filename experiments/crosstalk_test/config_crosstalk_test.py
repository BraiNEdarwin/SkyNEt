import numpy as np
from scipy import signal
from SkyNEt.config.config_class import config_class
import os

class experiment_config(config_class):


    def __init__(self):
        super().__init__() 

        self.waveElectrodes = 7
        self.factor = 0.05
        self.freq2 = np.array([2,np.pi,5,7,13,17,19]) 
        self.freq = np.sqrt(self.freq2[:self.waveElectrodes]) * self.factor
        self.phase = np.zeros(self.waveElectrodes)
        self.sampleTime = 50 # Initial sample time for slowest factor
        self.fs = 25
        c = 10
        #self.factor_gain = np.array([1,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,80,76,72,68,64,60,56,52,48,44,40,36,32,28,24,20,16,12,8,4,2,1]*4) # Increase in frequency
        self.factor_gain = np.array([1]*c + [2]*c + [4]*c + [8]*c + [12]*c + [16]*c + [20]*c + [24]*c + [28]*c + [32]*c + [36]*c + [40]*c + [44]*c + [48]*c + [52]*c + [56]*c + [60]*c \
            + [64]*c + [68]*c + [72]*c + [76]*c + [80]*c + [84]*c + [88]*c + [92]*c + [96]*c + [100]*c)

        #self.factor_gain = np.array([28]*c + [32]*c + [36]*c + [40]*c + [44]*c + [48]*c + [52]*c + [56]*c + [60]*c \
        #    + [64]*c + [68]*c + [72]*c + [76]*c + [80]*c + [84]*c + [88]*c + [92]*c + [96]*c + [100]*c)
        self.amplification = 100
        self.gain_info = '10MV/A'
        self.postgain = 1

        self.amplitude = 0.4*np.ones(7)
        self.offset = np.zeros(7)

        self.keithley_address = 'GPIB0::17::INSTR'
        #                               Summing module S2d      Matrix module           device
        self.electrodeSetup = [['ao5','ao3','ao1','ao0','a02','ao4','ao6','out'],[1,3,5,6,11,13,15,17],[5,6,7,8,1,2,3,4]]

        # Save settings
        self.filepath = r'D:\data\Mark\crosstalk_test\\'
        
        self.name = 'speedtest_50s_400mV_10x_ordered_fs_25'
        self.configSrc = os.path.dirname(os.path.abspath(__file__))        
        self.inputData = self.generateSineWave



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

