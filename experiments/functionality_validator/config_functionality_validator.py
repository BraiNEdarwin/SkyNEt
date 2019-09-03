import numpy as np
import os
from SkyNEt.config.config_class import config_class

class experiment_config(config_class):
    ''' 
    This experiment is used to validate functionality predicted by the NN.
    '''

    def __init__(self):
        super().__init__() #DO NOT REMOVE!
        ################################################
        ######### SPECIFY PARAMETERS ###################
        ################################################
        self.comport = 'COM3'  # COM port for the ivvi rack
        self.device = 'cDAQ'  # Either nidaq or adwin

        self.input_electrodes = [5,6] # Define which electrodes are used as inputs     
        self.controlVoltages = np.load(r'loadstring')['CV'] # given in volts (use load string or specify the 5 control voltages)
        #self.controlVoltages = np.array([-1.1714, 0.3669, 0.4937, 0.0138, -0.6853])[np.newaxis,:]
        self.inputScaling = 0.333593 #-1.0583    
        self.inputOffset = -np.array([-0.17727, 0.134817])
        
        # For the case of Boolean logic, define the 4 input cases (which are scaled with inputScaling and inputOffset)
        # For the ring, load the input data
        self.x = np.array([[0,0,1,1],[0,1,0,1]])* self.inputScaling + self.inputOffset
        
        # uncomment the following two for the ring input data
        #self.x = np.load(r'loadstring')['inp_wvfrm'].T 
        #self.x = self.x/np.max(np.abs(self.x)) * self.inputScaling + self.inputOffset[:,np.newaxis]
        # Define experiment
        
        self.postgain = 1
        self.amplification = 1000  # nA/V

        self.fs = 1000
        self.pointlength = 100   # Amount of datapoints for a single sample
        self.rampT = int(self.fs/100)    # datapoints to ramp from one datapoint to the next

        ################# Save settings ################
        self.filepath = r'D:\data\Mark\predict\\'
        self.configSrc = os.path.dirname(os.path.abspath(__file__))

        #                    Summing module S2d              Matrix module       on chip
        self.electrodeSetup = [['ao5','inp0','ao1''ao0','a02','inp4','ao6','out'],[1,3,5,6,11,13,15,17],[5,6,7,8,1,2,3,4]]
        self.name = 'name'

	
