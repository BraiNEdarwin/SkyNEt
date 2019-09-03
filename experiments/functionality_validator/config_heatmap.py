import numpy as np
from SkyNEt.config.config_class import config_class
import os

class experiment_config(config_class):
    '''

    '''

    def __init__(self):
        super().__init__() 

        self.waveElectrodes = 2
        self.inputIndex = [1,2]
        self.controlElectrodes = 5
        self.factor = 0.035
        self.freq2 = np.array([np.pi,5]) 
        self.freq = np.sqrt(self.freq2[:self.waveElectrodes])*self.factor
        self.phase = np.zeros(self.waveElectrodes)
        self.sampleTime = 1000 # Sample time of the sine waves for one grid point (in seconds)
        self.fs = 100

        self.samplePoints = int(50*self.fs) # Amount of sample points per batch measurement (sampleTime*fs/samplePoints batches)
        self.amplification = 100
        self.postgain = 1
        self.amplitude = np.array([0.75, 0.75]) # Maximum amount of voltage for the inputs
        self.offset = np.array([0., 0.]) # Offset for the sine waves
        self.controlVoltages = np.array([-0.9653, 0.1958, -0.3946, 0.2881, -0.5733])[np.newaxis,:] # XNOR
        #self.controlVoltages = np.load(r'D:\data\Mark\wave_search\paper_chip\2019_04_27_115357_train_data_2d_f_0_05\NN\ring\results_MSE_n_tanh_ring_inverse.npz')['CV']

        self.keithley_address = 'GPIB0::17::INSTR'
        #                               Summing module S2d      Matrix module           device
        self.electrodeSetup = [['ao5','ao3','ao1''ao0','a02','ao4','ao6','out'],[1,3,5,6,11,13,15,17],[5,6,7,8,1,2,3,4]]
        # Save settings
        self.filepath = r'D:\data\Mark\predict\paper_chip\heatmap\\'
        
        self.name = 'Heatmap_AND_1000s' 
        #self.name = 'speed_CV_Test_factor_' + str(self.factor) + '_T_' + str(self.sampleTime) + 's_batch_' + str(int(self.samplePoints/self.fs)) + 's'
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