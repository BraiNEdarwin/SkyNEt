import numpy as np
from SkyNEt.config.config_class import config_class
import os

class experiment_config(config_class):
    '''
    This is the config for the gradient descent experiment.
    
    '''

    def __init__(self):
        super().__init__() 
        
        self.device = 'chip' # Specifies whether the experiment is used on the NN or on the physical device. Is either 'chip' or 'NN'
        self.main_dir = r'NN_folder_string'
        self.NN_name = 'NN.pt'
        #######################
        # Physical parameters #
        #######################

        self.controls = 5
        self.inputs = 2
        self.freq = 8*np.array([1,2,3,4,5])  #
        self.fs = 1000
        self.n = 50               # Amount of iterations
        self.amplification = 100
        self.postgain = 1
        self.inputScaling = 1.8
        self.inputOffset = -1.2
        self.CVrange = np.array([[-1.2, .8],[-1.2, .8],[-1.2, -.8],[-0.8, 0.8],[-0.8, 0.8]])   # Range for the control voltages
        
        self.waveAmplitude = np.array([0.07, 0.03, 0.03, 0.005, 0.005])   # Amplitude of the waves used in the controls
        self.rampT = 0.3     # time to ramp up and ramp down the voltages at start and end of a measurement.
        self.targetGen = self.XOR
        self.name = 'name'
        #                        Summing module S2d      Matrix module           device
        # For the first array: 7 is always the output, 0 corresponds to ao0, 1 to ao1 etc.
        self.electrodeSetup = [[0,1,2,3,4,5,6,7],[1,3,5,7,11,13,15,17],[5,6,7,8,1,2,3,4]]
        
        self.controlLabels = ['ao0','ao1','ao2','ao3','ao4','ao5']
        self.inputIndex = [1,2] # Electrodes that will be used as boolean input
        
        ###################
        # rest parameters #
        ###################
        
        self.signallength = 0.125*4  #in seconds
        self.edgelength = 0.01  #in seconds
        self.inputCases = 4     #amount of cases measured (4 in the case of Boolean logic)
        
        self.phase_thres = 90 # in degrees
        self.eta = 3E-2          # Learn rate 
        self.gradFunct =  self.cor_sigmoid_grad
        self.errorFunct = self.cor_sigmoid_loss

        self.filepath =  r'filepath'    
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
        num = np.mean(x_min_m * t_min_m)     # numerator of corr
        denom = np.std(x) * np.std(t)        # denominator of corr    
        d_corr = ((t_min_m)/len(t_min_m) * denom - num * (x_min_m/len(x_min_m))/np.sqrt(np.mean(x_min_m**2)) * np.sqrt(np.mean(t_min_m**2))) / (denom**2)     
        return -d_corr # '-' sign because our corr is actually 1 - corr
    
    def cor_sigmoid_loss(self, x, t, w):
        x = x[w.astype(int)==1] # Remove all datapoints where w = 0
        t = t[w.astype(int)==1]
        corr = np.mean((x-np.mean(x))*(t-np.mean(t)))/(np.std(x)*np.std(t)+1E-12)
        x_high_min = np.min(x[(t == self.gainFactor)])
        x_low_max = np.max(x[(t == 0)])
        sigmoid = 1/(1 +  np.e**(-(x_high_min - x_low_max -5)/3)) + 0.05
        return (1.1 - corr) / sigmoid  
        
    def cor_sigmoid_grad(self, x, t, w):
        x = x[w.astype(int)==1] # Remove all datapoints where w = 0
        t = t[w.astype(int)==1]
        corr = np.mean((x-np.mean(x))*(t-np.mean(t)))/(np.std(x)*np.std(t)+1E-12)
        d_corr = self.cor_grad(x, t, w=np.ones(len(x)))  
        
        x_high_min = np.min(x[(t == self.gainFactor)])
        x_low_max = np.max(x[(t == 0)])
        
        sigmoid = 1/(1 +  np.e**(-(x_high_min - x_low_max -5)/3)) +0.05
        d_sigmoid = sigmoid*(1-sigmoid)
        
        return (d_corr * sigmoid - ((x == x_high_min).astype(int) - (x == x_low_max).astype(int)) * d_sigmoid * (1.1 - corr)) / sigmoid **2 
    
    def cor_sigmoid_grad2(self, x, t, w):
        x = x[w.astype(int)==1] # Remove all datapoints where w = 0
        t = t[w.astype(int)==1]

        d_corr = self.cor_grad(x, t, w=np.ones(len(x)))  
        
        x_high_min = np.min(x[(t == self.gainFactor)])
        x_low_max = np.max(x[(t == 0)])
        
        sigmoid = 1/(1 +  np.e**(-(x_high_min - x_low_max -5)/3)) +0.05

        
        return d_corr / sigmoid
    