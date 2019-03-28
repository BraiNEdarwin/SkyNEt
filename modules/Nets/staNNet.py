#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:40:12 2018

@author: hruiz

Defines the standard Neural Network class using PyTorch nn-package. It defines the model and inplements the training procedure. 
INPUT: ->data; a list of (input,output) pairs for training and validation [(x_train,y_train),(x_val,y_val)]. 
               The dimensions of x and y arrays must be (Samples,Dim) and they must be torch.FloatTensor
       ->depth: number of hidden layers
       ->width: can be list of number of nodes in each hidden layer
       ->kwarg: loss='MSE'; string name of loss function that defines the problem (only default implemented)
                activation='ReLU'; name of activation function defining the non-linearity in the network (only default implemented)
                betas=(0.9, 0.999) is a tuple with beta1 and beta2 for Adam
TO DO:
    - clean up the implementation of non linearity: it can be done by defining the non-lin at the object construction and so avoid the if-clauses
    - Apply BatchNorm
    - Implement a more flexible construction of the NN (with different widths for the hidden layers)
"""

import torch
import torch.nn as nn
import numpy as np 
#from matplotlib import pyplot as plt

class staNNet(object):
    
    def __init__(self,*args,loss='MSE',C=1.0,activation='ReLU', dim_cv=5, BN=False):
        
        self.ymax = 36 # Maximum value that the output can have (clipping value)
        self.C = torch.FloatTensor([C])
        
        if len(args) == 3: #data,depth,width
           data,depth,width = args
           self.x_train, self.y_train = data[0]
           self.x_val, self.y_val = data[1]
           self.D_in = self.x_train.size()[1]
           # how many of the inputs are control voltages
           self.dim_cv = dim_cv
           assert dim_cv < self.D_in, 'Total input dimension must be greater than cv dimension'
           self.D_out = self.y_train.size()[1]
           self._BN = BN
           self.depth = depth
           self.width = width
           self.loss_str = loss
           self.activ = activation
           self._tests()
        
           ################### DEFINE MODEL ######################################
           self._contruct_model()
           if isinstance(self.x_train.data,torch.cuda.FloatTensor): 
               self.itype = torch.cuda.LongTensor
               self.C.cuda()
               self.model.cuda()
               self.loss_fn.cuda()
               print('Sent to GPU')
           else: 
               self.itype = torch.LongTensor
            
        elif len(args)==1 and type(args[0]) is str:
            self._load_model(args[0])
            
        elif len(args) == 9:    #data,depth,width,freq,Vmax,fs,phase
           print('Input waves will be generated during training') 
           data,depth,width,self.freq,self.amplitude,self.fs,self.offset,self.phase,self.noisefit = args
           self.x_train, self.y_train = data[0]
           self.y_train = self.y_train/10 #\\
           self.x_val, self.y_val = data[1]
           self.y_val = self.y_val/10 #\\
           self.D_in = self.freq.shape[0]     
           self.dim_cv = dim_cv
           #assert dim_cv < self.D_in, 'Total input dimension must be greater than cv dimension'
           self.D_out = self.y_train.size()[1]
           self._BN = BN
           self.depth = depth
           self.width = width
           self.loss_str = loss
           self.activ = activation
           self._tests()
        
           ################### DEFINE MODEL ######################################
           self._contruct_model()
           if isinstance(self.x_train.data,torch.cuda.FloatTensor): 
               self.itype = torch.cuda.LongTensor
               self.C.cuda()
               self.model.cuda()
               self.loss_fn.cuda()
               print('Sent to GPU')
           else: 
               self.itype = torch.LongTensor
        else:
            assert False, 'Arguments must be either 3 (data,depth,width) or a string to load the model!'
            
        
    def _load_model(self,data_dir):
        print('Loading the model from '+data_dir)
        if torch.cuda.is_available():
            state_dic = torch.load(data_dir)
        else:
            state_dic = torch.load(data_dir, map_location='cpu')
        if list(filter(lambda x: 'running_mean' in x,state_dic.keys())):
            print('BN active in loaded model')
            self._BN = True
        else:
            self._BN = False
            
        self.loss_str = state_dic['loss']
        state_dic.pop('loss') #Remove entrie of OrderedDict
        
        self.activ = state_dic['activation']
        state_dic.pop('activation')  
          
        try:
            self.dim_cv = state_dic['dim_cv']
            state_dic.pop('dim_cv')
        except KeyError:
            self.dim_cv = 5
            print("Warning: Could not load attribute dim_cv, set at default 2.")
        try:
            #make a loop
            self.noisefit = state_dic['noise_model'] 
            state_dic.pop('noise_model')
            self.amplitude = state_dic['amplitude']
            state_dic.pop('amplitude')
            self.freq = state_dic['freq']
            state_dic.pop('freq')
            self.fs = state_dic['fs']
            state_dic.pop('fs')
            self.offset = state_dic['offset']
            state_dic.pop('offset')
            self.phase = state_dic['phase']
            state_dic.pop('phase')
        except KeyError:
            self.noisefit = False
            print("Sine wave input data not saved in model.")
        if self.noisefit:
            try:
                self.a = state_dic['noisefit_a'] 
                state_dic.pop('noisefit_a')
                self.b = state_dic['noisefit_b'] 
                state_dic.pop('noisefit_b')
            except KeyError:
                print('Noise fit parameters failed to load')
        
        print('NN loaded with activation ',self.activ,', loss ',self.loss_str, ' and cv dimension ', str(self.dim_cv))
        
        itms = list(state_dic.items())  
        layers = list(filter(lambda x: ('weight' in x[0]) and (len(x[1].shape)==2),itms))
        self.depth = len(layers)-2
        self.width = layers[0][1].shape[0]
        self.D_in = layers[0][1].shape[1]
        self.D_out = layers[-1][1].shape[0]

        self._contruct_model()
        
        self.model.load_state_dict(state_dic)

        if isinstance(layers[-1][1],torch.FloatTensor): 
            self.itype = torch.LongTensor
        else: 
            self.itype = torch.cuda.LongTensor
            self.C.cuda()
            self.model.cuda()
            self.loss_fn.cuda()    
        self.model.eval()
            
    def _contruct_model(self):
        # Use the nn package to define our model and loss function.
#        self.model = nn.Sequential(
#            nn.Linear(self.D_in, self.width),
#            nn.ReLU(),
#            nn.Linear(self.width, self.D_out),
#        )
        
        self.l_in = nn.Linear(self.D_in, self.width)
        self.l_out = nn.Linear(self.width, self.D_out)
        self.l_hid = nn.Linear(self.width,self.width)

        if self._BN: 
            track_running_stats=False  
            print('BN tracking average: ',track_running_stats)
            self.bn_layer = nn.BatchNorm1d(self.width,track_running_stats=track_running_stats)
       
        if self.activ == 'tanh':
            activ_func = nn.Tanh()
            print('Activation is tanh')            
        elif self.activ == 'ReLU':
            activ_func = nn.ReLU()
            print('Activation is ReLU')
        elif self.activ == None:
            activ_func = None
        else:
            assert False, 'Activation Function Not Recognized!'
        
        
        if self._BN: 
            modules = [nn.BatchNorm1d(self.D_in,track_running_stats=track_running_stats),
                       self.l_in,activ_func]
        else:
            modules = [self.l_in,activ_func]
            
        for i in range(self.depth):
            if self._BN: 
                modules.append(self.bn_layer) 
                #BN before activation (like in paper) doesn't make much difference; 
                #before the linearity also not much difference vs non-BN model
            modules.append(self.l_hid)
            modules.append(activ_func)
        
        modules.append(self.l_out)
        modules = [x for x in modules if x !=None ]
        
        print('Model constructed with modules: /n',modules)
        self.model = nn.Sequential(*modules)
        if self.noisefit == False:
            self.loss_fn = nn.MSELoss()
        else:
            # Fitting parameters for sigma = a*pred + b
            self.a = torch.tensor([-0.01553690797247516, 0.010176319709654977])
            self.b = torch.tensor([0.04678267320566001, 0.04678267320566001])
 
    def loss_fn(self, pred, targets):   
        sign = torch.sign(pred)
        sigma = -1 * (sign-1)/2 * (abs(self.a[0] * pred) + self.b[0]) + (sign+1)/2 * (abs(self.a[1] * pred) + self.b[1])              
        r = torch.mean(((pred - targets) ** 2) / sigma ** 2 ) 
        return r
   
    def train_nn(self,learning_rate,nr_epochs,batch_size,betas=(0.9, 0.999),seed=False):   
        """TO DO: 
            check if x_train, x_val and y_train and y_val are defined, if not, raise an error asking to define
        """
        
        if seed:
            torch.manual_seed(22)
            print('The torch RNG is seeded!')
            
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=betas) # OR SGD?!
        print('Prediction using ADAM optimizer')
        self.L_val = np.zeros((nr_epochs,))
        self.L_train = np.zeros((nr_epochs,))
        for epoch in range(nr_epochs):
    
            permutation = torch.randperm(self.x_train.size()[0]).type(self.itype) # Permute indices 
            running_loss = 0.0 
            nr_minibatches = 0

            for i in range(0,len(permutation),batch_size):
                
                # Forward pass: compute predicted y by passing x to the model.
                indices = permutation[i:i+batch_size]
                if self.x_train.shape[1] == 1:
                    y_pred = self.model(self.generateSineWave(self.freq, self.x_train[indices], self.amplitude, self.fs, self.offset, self.phase)) 
                else:
                    y_pred = self.model(self.x_train[indices]) 
                
                # Compute and print loss. 
                loss = self.loss_fn(y_pred, self.y_train[indices])
                #loss = loss*(self.C.cuda())
                running_loss += loss.item()      
                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable
                # weights of the model). This is because by default, gradients are
                # accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called. Checkout docs of torch.autograd.backward for more details.
                optimizer.zero_grad()
                
                # Backward pass: compute gradient of the loss with respect to model
                # parameters
                loss.backward()
                
                # Calling the step function on an Optimizer makes an update to its
                # parameters
                optimizer.step()
                nr_minibatches += 1
        
            
            self.model.eval()  
            
            if self.x_val.shape[1] == 1:
                y = self.model(self.generateSineWave(self.freq, self.x_val, self.amplitude, self.fs, self.offset, self.phase)) 
            else:
                y = self.model(self.x_val) 
            
            
            loss = self.loss_fn(y, self.y_val)
            self.model.train()  
            self.L_val[epoch] = loss.item()
            self.L_train[epoch] = running_loss/nr_minibatches
            print('Epoch:',epoch,'Val. Error:', loss.item(),'Training Error:',running_loss/nr_minibatches)
            
        print('Finished Training')
#        plt.figure()
#        plt.plot(np.arange(nr_epochs),self.L_val)
#        plt.show()
        
    def _tests(self):
        if not self.x_train.size()[0] == self.y_train.size()[0]: 
            raise NameError('Input and Output Batch Sizes do not match!')
    
    def save_model(self,path):
        self.model.eval()
        state_dic = self.model.state_dict()
        state_dic['activation'] = self.activ
        state_dic['loss'] = self.loss_str
        state_dic['dim_cv'] = self.dim_cv
        state_dic['noise_model'] = self.noisefit
        state_dic['freq'] = self.freq
        state_dic['amplitude'] = self.amplitude
        state_dic['offset'] = self.offset
        state_dic['fs'] = self.fs
        state_dic['phase'] = self.phase
        
        if self.noisefit:
            state_dic['noisefit_a'] = self.a
            state_dic['noisefit_b'] = self.b
        
        torch.save(state_dic,path)

    def outputs(self,inputs, *args):
        if len(args) == 5:
            freq,Vmax,fs,offset,phase = args
        if inputs.shape[1] == 1:
            return self.model(self.generateSineWave(freq,inputs,Vmax,fs,offset,phase)).data.cpu().numpy()[:,0]
        else:
            return self.model(inputs).data.cpu().numpy()[:,0]

      
    def generateSineWave(self,freq, t, amplitude, fs, offset = np.zeros(7), phase = np.zeros(7)):
        '''
        Generates a sine wave that can be used for the input data.

        freq:       Frequencies of the inputs in an one-dimensional array
        t:          The datapoint(s) index where to generate a sine value (1D array when multiple datapoints are used)
        amplitude:  Amplitude of the sine wave (Vmax in this case)
        fs:         Sample frequency of the device
        phase:      (Optional) phase offset at t=0
        '''
        
        waves = amplitude * np.sin((2 * np.pi * np.outer(t,freq) + phase)/ fs)  + np.outer(np.ones(t.shape[0]),offset)
        waves = torch.from_numpy(waves).type(torch.float32)
        return  waves    