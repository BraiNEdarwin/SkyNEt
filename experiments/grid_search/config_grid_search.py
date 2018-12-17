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

        self.controlVoltages =  [list(np.linspace(-900, 900, 37))]*3
        self.input4 = [-900, 900]
        self.input3 = [-900, 900]
        self.input2 = [-900, 900]
        self.input1 = [-900, 900]
        self.voltageGrid = [*self.controlVoltages,self.input4,self.input3,self.input2,self.input1]
        self.electrodes = len(self.voltageGrid)
        self.acqTime = 0.01
        self.samples = 50
        self.fs = 1000
        self.amplification = 1000
        self.postgain = 100

        self.keithley_address = 'GPIB0::17::INSTR'


        self.electrodeSetup = [[5,1,2,3,4,6,7,'out'],[1,3,5,7,11,13,15,17],[5,6,7,8,1,2,3,4]]
        # Save settings
        self.filepath = r'D:\\data\\Mark\\NN_grid\\'
        self.name = '4inp_3CV'
        self.configSrc = os.path.dirname(os.path.abspath(__file__))

    #####################################################
    ############# USER-SPECIFIC METHODS #################
    #####################################################
    # Optionally define new methods here that you wish to use in your experiment.
    # These can be e.g. new fitness functions or input/output generators.
