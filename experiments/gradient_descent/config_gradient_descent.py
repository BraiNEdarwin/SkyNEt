import numpy as np
from SkyNEt.config.config_class import config_class
import SkyNEt.experiments.gradient_descent.functionalities as fncts
import os

class experiment_config(config_class):
    '''
    This is the config for the gradient descent experiment.
    
    '''

    def __init__(self):
        super().__init__() 
        
        self.device = 'chip' # Specifies whether the experiment is used on the NN or on the physical device. Is either 'chip' or 'NN'
        self.main_dir = r'..\\..\\test\\'
        self.NN_name = 'checkpoint3000_02-07-23h47m.pt'
        
        self.fs = 5000
        self.edgelength = 0.03  #in seconds
        self.inputCases = 4     #amount of cases measured (4 in the case of Boolean logic)
        self.signallength = self.inputCases * 0.4 #(10/self.freq[0])*self.inputCases  # total signal in seconds (10 periods of slowest frequency)
        
        self.task = fncts.booleanLogic('XNOR',self.signallength,self.edgelength,self.fs) #fncts.featureExtractor(1,self.signallength,self.edgelength,self.fs) 
        
        self.name = 'XNOR_cor_sigmoid'
        
        self.verbose = True
        self.initializations = 1

        #######################
        # Physical parameters #
        #######################
        self.controls = 5
        self.inputs = 2
        self.freq = 5*np.array([5,9,13,19,23])  #  5*np.array([5,9,13])#    
        self.n = 50               # Amount of iterations
        self.amplification = 100
        self.postgain = 1
        self.inputScaling = 1.8
        self.inputOffset = -1.2
        self.CVrange =  np.array([[-1.2, .6],[-1.2, .6],[-1.2, .6],[-1.2, 0.6],[-1.2, 0.6]]) #np.array([[-1.2,0.6],[-0.8, 0.5],[-0.8,0.5]])   # Range for the control voltages        
        self.A_in = np.array([0.05, 0.02, 0.02, 0.01, 0.01])#np.array([0.07, 0.01, 0.01])#   # Amplitude of the waves used in the controls
              
        self.inputIndex = [1,2] # Electrodes that will be used as boolean input
        
        self.rampT = 0.3     # time to ramp up and ramp down the voltages at start and end of a measurement.
        self.phase_thres = 90 # in degrees 
        
        #############################
        # hyper parameters optimizer#
        #############################
        self.optimizer = self.basicGD
        self.beta_1 = 0.9
        self.beta_2 = 0.9 #0.75
        self.eta = 6E-2          # Learn rate  ~1E-3 for NMSE
        self.gradFunct =  self.cor_sigmoid_grad # self.NMSE_grad # 
        self.errorFunct = self.cor_sigmoid_loss # self.NMSE_loss # 
        
        ###################
        # rest parameters #
        ###################
                         
        #                        Summing module S2d      Matrix module           device
        # For the first array: 7 is always the output, 0 corresponds to ao0, 1 to ao1 etc.
        self.electrodeSetup = [[0,1,2,3,4,5,6,7],[1,3,5,6,11,13,15,17],[5,6,7,8,1,2,3,4]]
        
        self.controlLabels = ['ao5','ao3','ao1','ao0','ao2','ao4','out']


        self.filepath =  r'D:\data\Mark\GD\logic_gates\new_chip\\'    
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
        
        return sign * A_out/A_in

    ###########################################################################
    ############################### Optimizers ################################
    ###########################################################################
    
    def basicGD(self, step, controls, gradients, learn_rate, *args):
        # The '*args' is required for generality, e.g. the Adam optimizer contains more arguments, so in main code we need to fill in more args.
        # If we then switch to basicGD, the excessive arguments are stored in *args.
    
        return controls - learn_rate * gradients, 0, 0 # 0 are dummies for m and v, which are not used in basicGD
    
    def Adam(self, step, controls, gradients, learn_rate, m, v, beta_1 = 0.9, beta_2 = 0.999, eps = 10E-8):
        # Initialize first and second moment vector
        if step == 0:
            m = np.zeros(gradients.shape[0])
            v = np.zeros(gradients.shape[0])
        m = beta_1 * m + (1-beta_1) * gradients     # Update biased first moment estimate
        v = beta_2 * v + (1-beta_2) * gradients**2  # Update biased second raw moment estimate
        if step != 0:
            m = m/(1 - beta_1**step)   # Compute bias-corrected first moment estimate
            v = v/(1- beta_2**step)  # Compute bias-corrected second raw moment estimate
        controls = controls - learn_rate * m/(np.sqrt(v) + eps) # Update parameters            
        return controls, m, v
    
    ###########################################################################
    ############################# Cost functions ##############################
    ###########################################################################
    
    def MSE_loss(self,x,t,w):
        return np.sum(((x - t) * w)**2)/np.sum(w)

    def MSE_grad(self, x, t, w):
        ''' Calculates the mean squared error loss given the gradient of the 
        output w.r.t. the input voltages. This function calculates the error
        for each control separately '''      
        x = x[w.astype(int)==1] # Remove all datapoints where w = 0
        t = t[w.astype(int)==1]
        return 0.5 * (x - t) 
    
    def NMSE_loss(self,x,t,w):
        return np.sum(((x - t) * w)**2)/(np.sum(w) * (max(x)-min(x)))

    def NMSE_grad(self, x, t, w):
        ''' Calculates the normalized mean squared error loss given the gradient of the 
        output w.r.t. the input voltages. This function calculates the error
        for each control separately '''      
        x = x[w.astype(int)==1] # Remove all datapoints where w = 0
        t = t[w.astype(int)==1]
        return 0.5 * (x - t) / (max(x)-min(x))
        
        
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
   
    
    '''
    def cor_sigmoid_grad2(self, x, t, w):
        x = x[w.astype(int)==1] # Remove all datapoints where w = 0
        t = t[w.astype(int)==1]

        d_corr = self.cor_grad(x, t, w=np.ones(len(x)))  
        
        x_high_min = np.min(x[(t == self.gainFactor)])
        x_low_max = np.max(x[(t == 0)])
        
        sigmoid = 1/(1 +  np.e**(-(x_high_min - x_low_max -5)/3)) +0.05

        
        return d_corr / sigmoid   
    '''