#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 14:29:12 2018
@author: hruiz
"""

import numpy as np

class config_class(object):
    '''
    This class has all default parameter values and methods used in the 'standard' measurments 
    and Genetic Alg (GA), e.g. standard fitness functions and the input and target generators
    for the Boolean logic experiments.
    The class is used as a parent for the experiment_config() class that serves as a template for 
    all experiment configurations.
    The constructor __init__() contains all parameters that will be used in the GA and measurement
    script. Here, the fitness, target and inputs will be also defined. Their name should be 
    self.Fitness, self.Target_gen and self.Input_gen respectively. Notice that methods are written 
    with First_capital_letter to differentiate from parameters with small_letters. This convention should also be 
    carried to the user-specific experiment_config() classes.
    '''
    #TODO: clean unnecessary parameters
    #TODO: generalize code and clean up
    #TODO: rewrite parameters in small_letters and methods in First_capital_letter
    
    def __init__(self):
        ################################################
        ###### Config params for the experiments #######
        ################################################
        # Benchmarks settings
        self.benchmark = ['bl', 'XNOR']
          #'wr' for waveform regression benchmark
        self.WavePeriods = 15
        self.WaveFrequency = 8.5
        # WavePeriods2 = 0
        self.WaveFrequency2 = 18.5              
        # I/O settings
        self.SampleFreq = 1000
        self.skipstates = 0
        ################################################
        ############### Evolution settings #############
        ################################################
        self.genes = 5
        self.genomes = 25
        self.generations = 500
        self.generange = [[-900,900], [-900, 900], [-900, 900], [-900, 900], [-900, 900], [0.1, 2]]
        self.genelabels = ['CV1/T11','CV2/T13','CV3/T17','CV4/T7','CV5/T1', 'Input scaling']
        self.fitnessAvg = 1  #amount of runs per genome
        self.fitnessParameters = [1, 0, 1, 0.01]       
        ################################################
        ################# Save settings ################
        ################################################
        self.filepath = 'TEST_Evolution/'
        self.name = 'XNOR'
        
        ################################################
        ###################  Methods ###################
        ################################################
        self.Fitness = self.fitnessEvolution
        self.Target_gen = self.xor
        self.Input_gen = self.bool_input       
        # parameters for methods
        self.signalLength = 0.5  #in seconds 
        self.edgeLength = 0.01  #in seconds
        self.Fs = 1000
        
    #%%
    ####################################################
    ############# FITNESS METHODS ######################
    ####################################################
    def fitness_nmse(self, x, target):
        return 1 / ((np.linalg.norm(x - target, 2)) ** 2 * (1 / len(x)))

    def fitnessEvolution(self, x, target, W):
        '''This implements the fitness function
        F = self.fitnessParameters[0] * m / (sqrt(r) + self.fitnessParameters[3] * abs(c)) + self.fitnessParameters[1] / r + self.fitnessParameters[2] * Q
        where m,c,r follow from fitting x = m*target + c to minimize r
        and Q is the fitness quality as defined by Celestine in his thesis
        appendix 9'''
    
    
        #extract fit data with weights W
        indices = np.argwhere(W)  #indices where W is nonzero (i.e. 1)
    
        x_weighed = np.empty(len(indices))
        target_weighed = np.empty(len(indices))
        for i in range(len(indices)):
        	x_weighed[i] = x[indices[i]]
        	target_weighed[i] = target[indices[i]]
    
    
    	#fit x = m * target + c to minimize res
        A = np.vstack([target_weighed, np.ones(len(indices))]).T  #write x = m*target + c as x = A*(m, c)
        m, c = np.linalg.lstsq(A, x_weighed)[0]	
        res = np.linalg.lstsq(A, x_weighed)[1]
        res = res[0]
    
        #determine fitness quality
        indices1 = np.argwhere(target_weighed)  #all indices where target is nonzero
        x0 = np.empty(0)  #list of values where x should be 0
        x1 = np.empty(0)  #list of values where x should be 1
        for i in range(len(target_weighed)):
            if(i in indices1):
                x1 = np.append(x1, x_weighed[i])
            else:
                x0 = np.append(x0, x_weighed[i])
        if(min(x1) < max(x0)):
            Q = 0
        else:
            Q = (min(x1) - max(x0)) / (max(x1) - min(x0) + abs(min(x0)))
    
        F = self.fitnessParameters[0] * m / (res**(.5) + self.fitnessParameters[3] * abs(c)) + self.fitnessParameters[1] / res + self.fitnessParameters[2] * Q
        clipcounter = 0
        for i in range(len(x_weighed)):
            if(abs(x_weighed[i]) > 3.1*10):
                clipcounter = clipcounter + 1
                F = -100
        return F
    
    #%%
    ####################################################
    ############# OUTPUT METHODS #######################
    ####################################################
    def xor(self):
        samples = 4 * round(self.Fs * self.signalLength / 4) + 3 * round(self.Fs * self.edgeLength)
        x = np.empty(samples)
        t = np.linspace(0, samples/self.Fs, samples)
    
        x[0:round(self.Fs * self.signalLength / 4)] = 0
        x[round(self.Fs * self.signalLength / 4) : round(self.Fs * self.signalLength / 4) + round(self.Fs * self.edgeLength)] = np.linspace(1, 0, round(self.Fs * self.edgeLength))
        x[round(self.Fs * self.signalLength / 4) + round(self.Fs * self.edgeLength) : 2 * round(self.Fs * self.signalLength / 4) + round(self.Fs * self.edgeLength)] = 1
        x[2 * round(self.Fs * self.signalLength / 4) + round(self.Fs * self.edgeLength) : 2 * round(self.Fs * self.signalLength / 4) + 2 * round(self.Fs * self.edgeLength)] = 1
        x[2 * round(self.Fs * self.signalLength / 4) + 2 * round(self.Fs * self.edgeLength) : 3 * round(self.Fs * self.signalLength / 4) + 2 * round(self.Fs * self.edgeLength)] = 1
        x[3 * round(self.Fs * self.signalLength / 4) + 2 * round(self.Fs * self.edgeLength) : 3 * round(self.Fs * self.signalLength / 4) + 3 * round(self.Fs * self.edgeLength)] = 1
        x[3 * round(self.Fs * self.signalLength / 4) + 3 * round(self.Fs * self.edgeLength) : 4 * round(self.Fs * self.signalLength / 4) + 3 * round(self.Fs * self.edgeLength)] = 0
    
        return t, x
    
    #%%
    ####################################################
    ############# INPUT METHODS ########################
    ####################################################
    def bool_input(self):
        samples = 4 * round(self.Fs * self.signalLength / 4) + 3 * round(self.Fs * self.edgeLength)
        x = np.empty(samples)
        y = np.empty(samples)
        W = np.empty(samples)
        t = np.linspace(0, samples/self.Fs, samples)
    
        x[0:round(self.Fs * self.signalLength / 4)] = 0
        x[round(self.Fs * self.signalLength / 4) : round(self.Fs * self.signalLength / 4) + round(self.Fs * self.edgeLength)] = np.linspace(0, 1, round(self.Fs * self.edgeLength))
        x[round(self.Fs * self.signalLength / 4) + round(self.Fs * self.edgeLength) : 2 * round(self.Fs * self.signalLength / 4) + round(self.Fs * self.edgeLength)] = 1
        x[2 * round(self.Fs * self.signalLength / 4) + round(self.Fs * self.edgeLength) : 2 * round(self.Fs * self.signalLength / 4) + 2 * round(self.Fs * self.edgeLength)] = np.linspace(1, 0, round(self.Fs * self.edgeLength))
        x[2 * round(self.Fs * self.signalLength / 4) + 2 * round(self.Fs * self.edgeLength) : 3 * round(self.Fs * self.signalLength / 4) + 2 * round(self.Fs * self.edgeLength)] = 0
        x[3 * round(self.Fs * self.signalLength / 4) + 2 * round(self.Fs * self.edgeLength) : 3 * round(self.Fs * self.signalLength / 4) + 3 * round(self.Fs * self.edgeLength)] = np.linspace(0, 1, round(self.Fs * self.edgeLength))
        x[3 * round(self.Fs * self.signalLength / 4) + 3 * round(self.Fs * self.edgeLength) : 4 * round(self.Fs * self.signalLength / 4) + 3 * round(self.Fs * self.edgeLength)] = 1
    
        y[0:round(self.Fs * self.signalLength / 4)] = 0
        y[round(self.Fs * self.signalLength / 4) : round(self.Fs * self.signalLength / 4) + round(self.Fs * self.edgeLength)] = 0
        y[round(self.Fs * self.signalLength / 4) + round(self.Fs * self.edgeLength) : 2 * round(self.Fs * self.signalLength / 4) + round(self.Fs * self.edgeLength)] = 0
        y[2 * round(self.Fs * self.signalLength / 4) + round(self.Fs * self.edgeLength) : 2 * round(self.Fs * self.signalLength / 4) + 2 * round(self.Fs * self.edgeLength)] = np.linspace(0, 1, round(self.Fs * self.edgeLength))
        y[2 * round(self.Fs * self.signalLength / 4) + 2 * round(self.Fs * self.edgeLength) : 3 * round(self.Fs * self.signalLength / 4) + 2 * round(self.Fs * self.edgeLength)] = 1
        y[3 * round(self.Fs * self.signalLength / 4) + 2 * round(self.Fs * self.edgeLength) : 3 * round(self.Fs * self.signalLength / 4) + 3 * round(self.Fs * self.edgeLength)] = 1
        y[3 * round(self.Fs * self.signalLength / 4) + 3 * round(self.Fs * self.edgeLength) : 4 * round(self.Fs * self.signalLength / 4) + 3 * round(self.Fs * self.edgeLength)] = 1
    
        #define weight signal
        W[0:round(self.Fs * self.signalLength / 4)] = 1
        W[round(self.Fs * self.signalLength / 4) : round(self.Fs * self.signalLength / 4) + round(self.Fs * self.edgeLength)] = 0
        W[round(self.Fs * self.signalLength / 4) + round(self.Fs * self.edgeLength) : 2 * round(self.Fs * self.signalLength / 4) + round(self.Fs * self.edgeLength)] = 1
        W[2 * round(self.Fs * self.signalLength / 4) + round(self.Fs * self.edgeLength) : 2 * round(self.Fs * self.signalLength / 4) + 2 * round(self.Fs * self.edgeLength)] = 0
        W[2 * round(self.Fs * self.signalLength / 4) + 2 * round(self.Fs * self.edgeLength) : 3 * round(self.Fs * self.signalLength / 4) + 2 * round(self.Fs * self.edgeLength)] = 1
        W[3 * round(self.Fs * self.signalLength / 4) + 2 * round(self.Fs * self.edgeLength) : 3 * round(self.Fs * self.signalLength / 4) + 3 * round(self.Fs * self.edgeLength)] = 0
        W[3 * round(self.Fs * self.signalLength / 4) + 3 * round(self.Fs * self.edgeLength) : 4 * round(self.Fs * self.signalLength / 4) + 3 * round(self.Fs * self.edgeLength)] = 1
    
        W[round(self.Fs * self.signalLength / 4) + round(self.Fs * self.edgeLength): round(self.Fs * self.signalLength / 4) + round(self.Fs * self.edgeLength) + 40] = 0
        W[2 * round(self.Fs * self.signalLength / 4) + 2 * round(self.Fs * self.edgeLength): 2 * round(self.Fs * self.signalLength / 4) + 2 * round(self.Fs * self.edgeLength) + 40] = 0
        W[3 * round(self.Fs * self.signalLength / 4) + 3 * round(self.Fs * self.edgeLength): 3 * round(self.Fs * self.signalLength / 4) + 3 * round(self.Fs * self.edgeLength) + 40] = 0
        return t, x, y, W