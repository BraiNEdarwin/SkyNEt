import numpy as np
from SkyNEt.config.config_class import config_class

class experiment_config(config_class):
    '''
    This is the config file for switching the switch network setup
    to a particular device. There are actually only two parameters:

    comport; string, e.g. 'COM3' the comport to which the arduino is 
                connected
    device; int, 1-8 indicating the device to which you wish to switch
    '''

    def __init__(self):
        super().__init__() #DO NOT REMOVE!

        ################################################
        ######### SPECIFY PARAMETERS ###################
        ################################################
        self.comport = 'COM3'  # COM port of the arduino
        self.device = 1
		
		#define where you want to save the data.
        self.filepath = r'D:\Rik\IV\Device8\\'
        self.name = 'devicecheck'
        
        #define the impulse response you want to take in volts.
        self.v_off = 0
        self.v_pulse = 1
        self.n_points = 1000

        #define the input and output amplifications.
        self.amplification = 1
        self.source_gain = 1

        #measurment tool settings.
        self.device = 'adwin'
        self.fs = 1000
		
		#define feedback parameter
		self.W=0.8


        self.impulsegen = self.impulsegen

    def impulsegen(self, v_off, v_pulse, n_points):
        n_mid = n_points/2
        Input = np.full(n_points, v_off, float)
        Input[int(n_mid-1)]= v_pulse
        
        return Input
