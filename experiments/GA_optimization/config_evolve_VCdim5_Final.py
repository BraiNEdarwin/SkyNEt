#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:41:43 2019

@author: annefleur
"""

import numpy as np
from SkyNEt.config.config_class import config_class
from SkyNEt.modules.GenWaveform2 import GenWaveform


class experiment_config(config_class):
    '''This is the experiment configuration file to measure VC dim.
    This experiment_config class inherits from config_class default values that are known to work well with boolean logic.
    You can define user-specific parameters in the construction of the object in __init__() or define
    methods that you might need after, e.g. a new fitness function or input and output generators.
    Remember if you define a new fitness function or generator, you have to redefine the self.Fitness,
    self.Target_gen and self.Input_gen in __init__()
    ----------------------------------------------------------------------------
    Description of general parameters
    ----------------------------------------------------------------------------
    comport; the COM port to which the ivvi rack is connected.
    amplification; specify the amount of nA/V. E.g. if you set the IVVI to 100M,
        then amplification = 10
    generations; the amount of generations for the GA
    generange; the range that each gene (V) is mapped to. E.g. in the Boolean
        experiment the genes for the control voltages are mapped to the desired
        control voltage range.
    partition; this tells the GA how it will evolve the next generation.
        - copying the fittest partition[0] genomes to the next generation
        - the remaining partition[1] genomes are replaced by genomes resulting from crossover and mutation.
    genomes; the amount of genomes in the genepool
    genes; the amount of genes per genome
    fitnessavg; the amount of times the same genome is tested to obtain the fitness value.
    clipping value; the clipping value of the NN (after multiplication)
    ----------------------------------------------------------------------------
    Description of method parameters
    ----------------------------------------------------------------------------
    signallength; the length in s of the Boolean P and Q signals
    edgelength; the length in s of the edge between 0 and 1 in P and Q
    fs; sample frequency for niDAQ or ADwin
    ----------------------------------------------------------------------------
    Description of methods
    ----------------------------------------------------------------------------
    Fitness; specify the fitness function
    '''

    def __init__(self,inputs, labels):
        super().__init__() #DO NOT REMOVE!
        buf_str = str(labels)
        self.name = 'VCdim-'+''.join(buf_str.lstrip('[').strip(']').split(', '))
        self.name ='test' 
           
        ################################################
        ######### SPECIFY PARAMETERS ###################
        ################################################
        self.comport = 'COM3'  # COM port for the ivvi rack
        self.use_nn = False
        if self.use_nn:
            self.lengths, self.slopes = [100], [0] # in 1/fs
            self.amplification_nn = 10
        else:
            self.lengths, self.slopes = [100], [20] # in 1/fs
            self.amplification_chip = 100
        # Define experiment
        self.InputGen = self.input_waveform_new(inputs)
        self.TargetGen = np.asarray(GenWaveform(labels, self.lengths, slopes=self.slopes))
        self.generations = 80
        self.generange = [[-1.2,0.6], [-1.2,0.6],[-1.2,0.6], [-0.7,0.3], [-0.7,0.3]]
        self.inputrange = [-1.2,0.6]
        #electrodes of the input: [1,2], [3,4] or [5,6] 
        self.input_electrodes = [1,2]
        self.input_scaling = 1.0
        self.partition = [4,22]
        self.genomes = sum(self.partition)  
        self.genes = len(self.generange) 
        self.clipvalue = 350
        self.Fitness = self.corr_lin_fit
#       self.Fitness = self.corr_sig_fit

    


    #####################################################
    ############# USER-SPECIFIC METHODS #################
    #####################################################
    # Optionally define new methods here that you wish to use in your experiment.
    # These can be e.g. new fitness functions or input/output generators.

    def input_waveform(self, inputs):
        assert len(inputs) == 2, 'Input must be 2 dimensional!'
        inp_wvfrm0 = GenWaveform(inputs[0], self.lengths, slopes=self.slopes)
        inp_wvfrm1 = GenWaveform(inputs[1], self.lengths, slopes=self.slopes)
        samples = len(inp_wvfrm0)
        time_arr = np.linspace(0, samples/self.fs, samples)
        inputs_wvfrm = np.asarray([inp_wvfrm0,inp_wvfrm1])
        
#        print('Size of input', inputs_wvfrm.shape)
        w_ampl = [1,0]*len(inputs[0])
        w_lengths = [self.lengths[0],self.slopes[0]]*len(inputs[0])
        
        weight_wvfrm = GenWaveform(w_ampl, w_lengths)
        bool_weights = [x==1 for x in weight_wvfrm[:samples]]
        
        return time_arr, inputs_wvfrm, bool_weights
    
    def input_waveform_new(self, inputs):
        assert len(inputs) == 2, 'Input must be 2 dimensional!'
        inp_wvfrm0 = GenWaveform(inputs[0], self.lengths, slopes=self.slopes)
        inp_wvfrm1 = GenWaveform(inputs[1], self.lengths, slopes=self.slopes)
        samples = len(inp_wvfrm0)
        time_arr = np.linspace(0, samples/self.fs, samples)
        inputs_wvfrm = np.asarray([inp_wvfrm0,inp_wvfrm1])
        
#        print('Size of input', inputs_wvfrm.shape)
        w_ampl = [0] + [1,0]*len(inputs[0])
        w_lengths = [self.slopes[0]]+ [self.lengths[0],self.slopes[0]]*len(inputs[0])
        
        weight_wvfrm = GenWaveform(w_ampl, w_lengths)
        bool_weights = [x==1 for x in weight_wvfrm[:samples]]
        return time_arr, inputs_wvfrm, bool_weights

#    --------------------------------------------------------------------------------------
    #Fitness function1: Combination of a sigoid with pre-defined separation threshold (2.5 nA)
    # and the correlation function. The sigmoid can be adapted by changing the function 'sig(self, x)'        
    def corr_sig_fit(self, output, target, w):
        if np.any(np.abs(output[w])>self.clipvalue):
            print('Clipping value set at: '+ str(self.clipvalue))
            fit = -100
            return fit
        elif np.any(np.abs(output[w])<-self.clipvalue):
            print('Clipping value set at:'+ str(-self.clipvalue))
            fit = -100
            return fit
        buff0 = target[w] == 0
        buff1 = target[w] == 1
        max_0 = np.max(output[w][buff0])
        min_1 = np.min(output[w][buff1])
        sep = min_1 - max_0
        x = output[w][:,np.newaxis]
        y = target[w][:,np.newaxis]
        X = np.stack((x, y), axis=0)[:,:,0]
        corr = np.corrcoef(X)[0,1]
        #if sep > 0 it is linearly separable and we can break in the evolve script
        # this is done by evaluating whether the threshold fit is bigger than 0.1
        if sep >= 0: 
            fit =self.sig(sep) * corr
        else:
            fit = self.sig(sep) * corr * 0.01
        return fit
    
    #Sigmoid function. 
    def sig(self, sep):
        return 1/(1+np.exp(-5*(sep/2.5-0.5)))+ 0.1
    
#    --------------------------------------------------------------------------------------   
        
    #Fitness function2: Combination of a linear function with output-dependent thresholds 
    # and the correlation function. 
    # The linear function can be adapted by changing 'sig_lin(self,sep,standard_dev) 
    def corr_lin_fit(self, output, target, w):
        standard_deviation = [] 
        for k in range(0,len(output[w]),self.lengths[0]):
            standard_deviation.append(np.std(output[w][k:k+self.lengths[0]])) 
        standard_dev = np.asarray(standard_deviation)
        if np.any(np.abs(output[w])>self.clipvalue):
            print('Clipping value set at: '+ str(self.clipvalue))
            fit = -100
            return fit
        elif np.any(np.abs(output[w])<-self.clipvalue):
            print('Clipping value set at:'+ str(-self.clipvalue))
            fit = -100
            return fit
        buff0 = target[w] == 0
        buff1 = target[w] == 1
        max_0 = np.max(output[w][buff0])
        min_1 = np.min(output[w][buff1])
        sep = min_1 - max_0
        x = output[w][:,np.newaxis]
        y = target[w][:,np.newaxis]
        X = np.stack((x, y), axis=0)[:,:,0]
        corr = np.corrcoef(X)[0,1]
        #if sep > 0 it is linearly separable and we can break in the evolve script
        # this is done by evaluating whether the threshold fit is bigger than 0.1
        if sep >= 0: 
            corr =self.lin(sep, standard_dev) * corr
        else:
            corr = self.lin(sep, standard_dev) * corr * 0.01
        return corr 
        
    #Linear function with thresholds
    #For more information about the 
    #More focus can be put on separation by changing the prefactor in x_sep    
    def lin(self, sep, standard_dev):
        max_std = np.amax(standard_dev)
        second_std = np.amax(standard_dev[standard_dev!=max_std])
        #Threshold separation 
        x_sep = 4*(max_std+ second_std)
        #Start of the linear increase
        start = -4
        #If the threshold value x_sep is reached, 'maximum' will be returned. 
        maximum = 4
        if sep < start:
            return 0.1
        elif sep > x_sep:
            return maximum
        else: 
            rico = (maximum-0.1)/(x_sep-start)
            return rico*(sep-x_sep) + maximum
            
