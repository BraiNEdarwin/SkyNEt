import numpy as np
from SkyNEt.config.config_class import config_class
import os

class experiment_config(config_class):
    '''
    This is the config for the grid search experiment.

    ----------------------------------------------------------------------------
    Description of general parameters
    ----------------------------------------------------------------------------
    filepath = 'D:\data\data4nn'
    name = 'FullSwipe_TEST'
    controlVoltages = [[-900, -600, -300, 0, 300, 600, 900]]*5
    input2 = [-900, -600, -300, 0, 300, 600, 900]
    input1 = [-900,0,900]
    voltageGrid = [*controlVoltages,input2,input1]
    electrodes = len(voltageGrid) #amount of electrodes
    acqTime = 0.01
    samples = 50

    '''

    def __init__(self):
        super().__init__() #DO NOT REMOVE!
        ################################################
        ######### SPECIFY PARAMETERS ###################
        ################################################

        # Specify CVs as list of lists.
        #self.controlVoltages = [[-750, -450, -150, 0, 150, 450, 750]]*5
        #self.input2 = [-900, -600, -300, 0, 300, 600, 900]
        #self.input1 = [-900,0,900]

        self.waveElectrodes = 7

        self.factor = 2
        self.freq2 = np.array([2,np.pi,5,7,13,17,19]) # 
        self.freq = np.sqrt(self.freq2[:self.waveElectrodes])*self.factor
        self.sampleTime = 1000 # Sample time of the sine waves for one grid point (in seconds)

        self.fs = 500
        self.transientTest = False
        self.n = 200
        
        # If the input data is too large for RAM, load the data in small parts
        self.loadData = True
        self.loadPoints = 50000
        self.batches = 10 # 1720 = 23.9h for f = 500, loadpoints = 25000
        self.loadString = r'D:\data\Mark\wave_search\inputs\inputData_f2_24h_500Hz.npz'
        
        self.amplification = 1
        self.postgain = 1
        self.Vmax = 0.8 # Maximum amount of voltage for the inputs

        self.keithley_address = 'GPIB0::17::INSTR'

        #                               Summing module S2d      Matrix module           device
        self.electrodeSetup = [['ao0','ao2','ao4''ao6','a05','ao3','ao1','out'],[1,3,5,7,11,13,15,17],[5,6,7,8,1,2,3,4]]
        # Save settings
        self.filepath = r'D:\\data\\Mark\\wave_search\\'
        #self.name = 'wave_search_f'+str(self.factor) + 'sampleTime_' + str(int(self.sampleTime)) + 's_loadEvery_' + str(int(self.loadPoints/self.fs)) + 's'
        self.name = 'Sync_load_test_10x100s'
        self.configSrc = os.path.dirname(os.path.abspath(__file__))
        
        
        self.inputData = self.generateSineWave


    #####################################################
    ############# USER-SPECIFIC METHODS #################
    #####################################################
    # Optionally define new methods here that you wish to use in your experiment.
    # These can be e.g. new fitness functions or input/output generators.

    def generateSineWave(self, freq, t, amplitude, fs, phase = np.zeros(7)):
        '''
        Generates a sine wave that can be used for the input data.

        freq:       Frequencies of the inputs in an one-dimensional array
        t:          The datapoint(s) index where to generate a sine value (1D array when multiple datapoints are used)
        amplitude:  Amplitude of the sine wave (Vmax in this case)
        fs:         Sample frequency of the device
        phase:      (Optional) phase offset at t=0
        '''
        return np.sin((2 * np.pi * freq[:, np.newaxis] * t + phase)/ fs) * amplitude
