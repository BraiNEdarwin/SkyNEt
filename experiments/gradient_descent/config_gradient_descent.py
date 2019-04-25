import numpy as np
from SkyNEt.config.config_class import config_class
import os

class experiment_config(config_class):
    '''
    This is the config for the gradient descent experiment.
    
    '''

    def __init__(self):
        super().__init__() 
        
        self.device = 'NN' # Specifies whether the experiment is used on the NN or on the physical device.
        self.main_dir = r'C:\Users\User\APH\Thesis\Data\wave_search\champ_chip\2019_04_05_172733_characterization_2days_f_0_05_fs_50\nets\MSE_n_proper\\'
        self.NN_name = 'MSE_n_d10w90_300ep_lr3e-3_b1024_b1b2_0.90.75_seed.pt'
        #######################
        # Physical parameters #
        #######################

        self.controls = 5
        self.inputs = 2
        self.freq = 0.05*np.array([5, 3, 7, 11, 13])  #
        self.fs = 1000
        self.n = 50               # Amount of iterations
        self.amplification = 1000
        self.postgain = 100
        self.inputScaling = 0.9
        self.inputOffset = -0.5
        self.CVrange = np.array([[-0.8, 0.2],[-0.8, 0.2],[-1.1, 0.8],[-1.1, 0.8],[-1.1, 0.8]])   # Range for the control voltages
        
        self.waveAmplitude = 0.005    # Amplitude of the waves used in the controls
        self.rampT = 0.5           # time to ramp up and ramp down the voltages at start and end of a measurement.
        self.targetGen = self.XNOR
        self.name = 'NN_XNOR'
        #                        Summing module S2d      Matrix module           device
        # For the first array: 7 is always the output, 0 corresponds to ao0, 1 to ao1 etc.
        self.electrodeSetup = [[0,1,2,3,4,5,6,7],[1,3,5,7,11,13,15,17],[5,6,7,8,1,2,3,4]]
        
        self.controlLabels = ['ao0','ao1','ao2','ao3','ao4','ao5']
        self.inputIndex = [2,3] # Electrodes that will be used as boolean input
        
        ###################
        # rest parameters #
        ###################
        # parameters for methods
        self.signallength = 80  #in seconds
        self.edgelength = 0.01  #in seconds
        self.inputCases = 4     #amount of cases measured (4 in the case of Boolean logic)
        
        self.fft_N = self.fs*self.signallength//self.inputCases       
        self.phase_thres = 0.8
        self.eta = 3E-2           # Learn rate 
        self.gradFunct = self.cor_grad #self.MSE_grad
        self.errorFunct = self.cor_loss #self.MSE_loss
        self.keithley_address = 'GPIB0::17::INSTR'
        # Save settings
        #self.filepath = r'D:\data\\Mark\gradient_descent\\'
        self.filepath =  r'C:\Users\User\APH\Thesis\Data\gradient_descent\\'
    
        self.configSrc = os.path.dirname(os.path.abspath(__file__))
        self.gainFactor = self.amplification/self.postgain

    def MSE_loss(self,x,t,w):
        return np.sum(((x - t) * w)**2)/np.sum(w)

    def MSE_grad(self, x, t, w):
        ''' Calculates the mean squared error loss given the gradient of the 
        output w.r.t. the input voltages. This function calculates the error
        for each control separately '''      
        x = x[w.astype(int)==1] # Remove all datapoints where w = 0
        t = t[w.astype(int)==1]
        return 0.5 * (x - t) 
        
    def cor_separation_loss(self, x, t, w):
        x = x[w.astype(int)==1] # Remove all datapoints where w = 0
        t = t[w.astype(int)==1]
        
        corr = np.mean((x-np.mean(x))*(t-np.mean(t)))/(np.std(x)*np.std(t)+1E-12)
        x_high_min = np.min(x[(t == self.gainFactor)])
        x_low_max = np.max(x[(t == 0)])
        return (1.0001 - corr)/(abs(x_high_min-x_low_max)/2)**.5
    
    def cor_separation_grad(self, x, t, w):
        x = x[w.astype(int)==1] # Remove all datapoints where w = 0
        t = t[w.astype(int)==1]      
        x_min_m = x - np.mean(x)
        t_min_m = t - np.mean(t)       
                    
        d_corr = (t_min_m)/(np.std(x)*np.std(t) + 1E-12) - np.mean(x_min_m * t_min_m)* (x_min_m) / (np.std(t) * np.std(x)**3)
        
        # separation 
        x_high_min = np.min(x[(t == self.gainFactor)])
        x_low_max = np.max(x[(t == 0)])
        return -d_corr/(abs(x_high_min-x_low_max)/2)**.5 
    
    
    def cor_loss(self, x, t, w):
        x = x[w.astype(int)==1] # Remove all datapoints where w = 0
        t = t[w.astype(int)==1]
        
        corr = np.mean((x-np.mean(x))*(t-np.mean(t)))/(np.std(x)*np.std(t)+1E-12)
        return 1.00 - corr
    
    def cor_grad(self, x, t, w):
        x = x[w.astype(int)==1] # Remove all datapoints where w = 0
        t = t[w.astype(int)==1]
        x_min_m = x - np.mean(x)
        t_min_m = t - np.mean(t)
           
        d_corr = (t_min_m)/(np.std(x)*np.std(t) + 1E-12) - np.mean(x_min_m * t_min_m)* (x_min_m) / (np.std(t) * np.std(x)**3)      
        return -d_corr
    
    