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

        self.controls = 5
        self.freq = np.array([10, 12, 14, 16, 18])  #
        self.fs = 1000
        self.n = 20                 # Amount of iterations
        self.amplification = 1000
        self.postgain = 100
        self.Vmax = 0.9 # Maximum voltage for the inputs
        self.waveAmplitude = 0.1 # Amplitude of the waves used in the controls

        self.keithley_address = 'GPIB0::17::INSTR'#                              Summing module S2d      Matrix module           device
        self.electrodeSetup = [['ao0','ao2','ao4''ao6','a05','ao3','ao1','out'],[1,3,5,7,11,13,15,17],[5,6,7,8,1,2,3,4]]
        # Save settings
        self.filepath = r'D:\\data\\Mark\\gradient_descent\\'
        self.name = 'gradient_descent'
        self.configSrc = os.path.dirname(os.path.abspath(__file__))
        

    #####################################################
    ############# USER-SPECIFIC METHODS #################
    #####################################################
    # Optionally define new methods here that you wish to use in your experiment.
    # These can be e.g. new fitness functions or input/output generators.
