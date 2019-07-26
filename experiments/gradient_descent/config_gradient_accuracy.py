import numpy as np
from SkyNEt.config.config_class import config_class
import os

class experiment_config(config_class):
    '''
    This is the config for the gradient accuracy experiment.
    
    '''

    def __init__(self):
        super().__init__() 
        

        self.device = 'chip' # Specifies whether the experiment is used on the NN or on the physical device.
        self.main_dir = r'C:\Users\User\APH\Thesis\Data\wave_search\paper_chip\2019_04_27_115357_train_data_2d_f_0_05\NN_new\RMSE\\'
        self.NN_name = 'checkpoint770.pt'
        #######################
        # Physical parameters #
        #######################

        self.controls = 7
        self.factors = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
        self.signallength = 1  #in seconds 
        self.staticControls = 0.6*np.array([1,1,1,1,1,1,1])
        self.freq = np.array([1,2,3,4,5,6,7])  #
        self.fs = 1000
        self.n = 20               # Amount of iterations

        self.amplification = 100
        self.postgain = 1
        
        self.waveAmplitude = np.array([0.07, 0.05, 0.05, 0.03, 0.03, 0.005, 0.005])   # Amplitude of the waves used in the controls
        self.rampT = 0.5           # time to ramp up and ramp down the voltages at start and end of a measurement.
        self.name = '600mV_f_1-15_n_20_t_1'
        #                        Summing module S2d      Matrix module           device
        # For the first array: 7 is always the output, 0 corresponds to ao0, 1 to ao1 etc.
        self.electrodeSetup = [[0,1,2,3,4,5,6,7],[1,3,5,7,11,13,15,17],[5,6,7,8,1,2,3,4]]
        
        self.controlLabels = ['ao0','ao1','ao2','ao3','ao4','ao5']
        
        ###################
        # rest parameters #
        ###################
          
        self.phase_thres = 90 # in degrees
        self.keithley_address = 'GPIB0::17::INSTR'
        # Save settings
        #self.filepath = r'D:\data\\Mark\gradient_descent\\'
        self.filepath =  r'D:\data\Mark\gradient_accuracy\paper_chip\\'
    
        self.configSrc = os.path.dirname(os.path.abspath(__file__))
        self.gainFactor = self.amplification/self.postgain

