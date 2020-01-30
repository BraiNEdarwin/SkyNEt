import numpy as np
from SkyNEt.config.config_class import config_class
import os

class experiment_config(config_class):
    '''
    This is the config for testing the accuracy of multiwave versus single wave
    
    '''
    def __init__(self):
        super().__init__() 
        
        self.device = 'chip' # Specifies whether the experiment is used on the NN or on the physical device. Is either 'chip' or 'NN'
        self.main_dir = r'..\\..\\test\\'
        self.NN_name = 'checkpoint3000_02-07-23h47m.pt'
        self.verbose = True
        #######################
        # Physical parameters #
        #######################

        self.controls = 7
        self.factors = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.periods = 10        
        self.staticControls = np.random.random(self.controls)*1.8 - 1.2 #

        self.freq = np.array([2,5,3,9,13,19,23])[::-1]  # np.array([2,4,3,5,6,7,8]) #
        self.fs = 2000
        self.n = 10               # Amount of iterations

        self.amplification = 100
        self.postgain = 1
        
        self.waveAmplitude = 0.01*np.array([4,3,3,2,2,1,1])#np.array([0.07, 0.05, 0.05, 0.03, 0.03, 0.005, 0.005])   # Amplitude of the waves used in the controls
        self.rampT = 0.5           # time to ramp up and ramp down the voltages at start and end of a measurement.
        self.name = 'CV_rand_p10_f_1-10_fs_2000_f_2-23Hz_inverse_f2_f3_inverse_freqs_low_amp'
        #                        Summing module S2d      Matrix module           device
        # For the first array: 7 is always the output, 0 corresponds to ao0, 1 to ao1 etc.
        self.electrodeSetup = [[0,1,2,3,4,5,6,7],[1,3,5,6,11,13,15,17],[5,6,7,8,1,2,3,4]]
        
        self.controlLabels = ['ao5','ao3','ao1','ao0','ao2','ao4', 'ao6','out']
        
        ###################
        # rest parameters #
        ###################
          
        self.phase_thres = 90 # in degrees
        self.filepath =  r'D:\\data\\Mark\\GD\\multiwave_accuracy\new_chip\\'
    
        self.configSrc = os.path.dirname(os.path.abspath(__file__))
        self.gainFactor = self.amplification/self.postgain
        
        
    def lock_in_gradient(self, output, freq, A_in, fs=1000, phase_thres=90): 
        ''' This function calculates the gradients of the output with respect to
        the given frequencies using the lock-in method and outputs them in the 
        same order as the frequencies are given.
        output:         output data to compute derivative for
        freq:           frequencies for the derivatives
        A_in:           amplitude used for input waves
        fs:             sample frequency
        phase_thres:    threshold for the phase of the output wave, determines whether the gradient is positive or negative
        '''
        
        t = np.arange(0,output.shape[0]/fs,1/fs)
        y_ref1 = np.sin(freq[:,np.newaxis] * 2*np.pi*t)
        y_ref2 = np.sin(freq[:,np.newaxis] * 2*np.pi*t + np.pi/2)
        
        y_out1 = y_ref1 * (output - np.mean(output))
        y_out2 = y_ref2 * (output - np.mean(output))
        
        amp1 = (np.mean(y_out1,axis=1)) # 'Integrating' over the multiplied signal
        amp2 = (np.mean(y_out2,axis=1))
        
        A_out = 2*np.sqrt(amp1**2 + amp2**2)
        phase_out = np.arctan2(amp2,amp1)*180/np.pi
        sign = 2*(abs(phase_out) < phase_thres) - 1
        
        return sign * A_out/A_in, phase_out
