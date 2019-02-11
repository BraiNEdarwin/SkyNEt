import numpy as np
from SkyNEt.config.config_class import config_class

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
        self.controlVoltages = [[-500, 500]]*5
        self.input2 = [-500, 500]
        self.input1 = [-500, 500]
        self.voltageGrid = [*self.controlVoltages,self.input2,self.input1]
        self.electrodes = len(self.voltageGrid)
        self.acqTime = 0.01
        self.samples = 50

        # Save settings
        self.filepath = r'D:\data\path\to\your\directory\\'
        self.name = 'AND'

    #####################################################
    ############# USER-SPECIFIC METHODS #################
    #####################################################
    # Optionally define new methods here that you wish to use in your experiment.
    # These can be e.g. new fitness functions or input/output generators.
