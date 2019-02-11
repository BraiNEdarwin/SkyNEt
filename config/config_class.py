#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 14:29:12 2018
@author: hruiz
"""

import numpy as np
import signal
import sys

class config_class(object):
    '''
    This class has all default parameter values and methods used in the 'standard' measurements
    and Genetic Alg (GA), e.g. standard fitness functions and the input and target generators
    for the Boolean logic experiments.
    The class is used as a parent for the experiment_config() class that serves as a template for
    all experiment configurations.
    The constructor __init__() contains all parameters that will be used in the GA and measurement
    script. Here, the fitness, target and inputs will be also defined. Their name should be
    self.Fitness, self.TargetGen and self.InputGen respectively. Notice that methods are written
    with CamelCase to differentiate from parameters with smallletters. This convention should also be
    carried to the user-specific experiment_config() classes.

    ----------------------------------------------------------------------------
    Description of general parameters
    ----------------------------------------------------------------------------
    generations; the amount of generations for the GA

    generange; the range that each gene ([0, 1]) is mapped to. E.g. in the Boolean
        experiment the genes for the control voltages are mapped to the desired
        control voltage range.

    default_partition; the default setting for partition (see next)

    partition; this tells the GA how it will evolve the next generation.
        In order, this will make the GA evolve the specified number with
        - promoting fittest partition[0] genomes unchanged
        - adding Gaussian noise to the fittest partition[1] genomes
        - crossover between fittest partition[2] genomes
        - crossover between fittest partition[3] and randomly selected genes
        - randomly adding parition[4] genomes

    genomes; the amount of genomes in the genepool

    genes; the amount of genes per genome

    mutationrate; the probability of mutation for each gene (between 0 and 1)

    fitnessavg; the amount of times the same genome is tested to obtain the fitness
        value.

    fitnessparameters; the parameters for FitnessEvolution (see its doc for
        specifics)

    ----------------------------------------------------------------------------
    Description of method parameters
    ----------------------------------------------------------------------------
    signallength; the length in s of the Boolean P and Q signals
    edgelength; the length in s of the edge between 0 and 1 in P and Q
    fs; sample frequency for niDAQ or ADwin

    ----------------------------------------------------------------------------
    For a description of method, refer to their individual docstrings.

    '''

    def __init__(self):
        ################################################
        ###### Config params for the experiments #######
        ################################################
        self.fs = 1000
        self.comport = 'COM3'  # COM port for the ivvi rack

        ################################################
        ############### Evolution settings #############
        ################################################
        self.generations = 500
        self.generange = [[-600,600], [-900, 900], [-900, 900], [-900, 900], [-600, 600], [0.1, 0.5]]
        self.default_partition = [5, 5, 5, 5, 5]
        self.partition = self.default_partition.copy()
        self.genomes = 25
        self.genes = 6
        self.mutationrate = 0.1
        self.fitnessavg = 1
        self.fitnessparameters = [1, 0, 1, 0.01]

        ################################################
        ###################  Methods ###################
        ################################################
        self.Fitness = self.FitnessEvolution
        self.InputGen = self.BoolInput

        # parameters for methods
        self.signallength = 0.5  #in seconds
        self.edgelength = 0.01  #in seconds


    #%%
    ####################################################
    ############# FITNESS METHODS ######################
    ####################################################
    def FitnessNMSE(self, x, target):
        '''
        This function returns the normalized mean squared error of x w.r.t. target.
        '''
        return 1 / ((np.linalg.norm(x - target, 2)) ** 2 * (1 / len(x)))

    def FitnessEvolution(self, x, target, W):
        '''
        This implements the fitness function
        F = self.fitnessparameters[0] * m / (sqrt(r) + self.fitnessparameters[3] * abs(c)) + self.fitnessparameters[1] / r + self.fitnessparameters[2] * Q
        where m,c,r follow from fitting x = m*target + c to minimize r
        and Q is the fitness quality as defined by Celestine in his thesis
        appendix 9
        W is a weight array consisting of 0s and 1s with equal length to x and
        target. Points where W = 0 are not included in fitting.
        '''

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

        F = self.fitnessparameters[0] * m / (res**(.5) + self.fitnessparameters[3] * abs(c)) + self.fitnessparameters[1] / res + self.fitnessparameters[2] * Q
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
    def XOR(self):
        '''Returns an XOR signal with time array t'''
        samples = 4 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)
        x = np.empty(samples)
        t = np.linspace(0, samples/self.fs, samples)

        x[0:round(self.fs * self.signallength / 4)] = 0
        x[round(self.fs * self.signallength / 4) : round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength)] = np.linspace(1, 0, round(self.fs * self.edgelength))
        x[round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength) : 2 * round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength)] = 1
        x[2 * round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength) : 2 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength)] = 1
        x[2 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength) : 3 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength)] = 1
        x[3 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength) : 3 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)] = 1
        x[round(self.fs * self.signallength / 4) : round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength)] = np.linspace(0, 1, round(self.fs * self.edgelength))
        x[3 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength) : 4 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)] = 0

        return t, x

    def AND(self):
        '''Returns an AND signal with time array t'''
        samples = 4 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)
        x = np.empty(samples)
        t = np.linspace(0, samples/self.fs, samples)

        x[0:round(self.fs * self.signallength / 4)] = 0
        x[round(self.fs * self.signallength / 4) : round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength)] = 0
        x[round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength) : 2 * round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength)] = 0
        x[2 * round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength) : 2 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength)] = 0
        x[2 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength) : 3 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength)] = 0
        x[3 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength) : 3 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)] = np.linspace(0, 1, round(self.fs * self.edgelength))
        x[3 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength) : 4 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)] = 1

        return t, x

    def OR(self):
        '''Returns an OR signal with time array t'''
        samples = 4 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)
        x = np.empty(samples)
        t = np.linspace(0, samples/self.fs, samples)

        x[0:round(self.fs * self.signallength / 4)] = 0
        x[round(self.fs * self.signallength / 4) : round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength)] = np.linspace(0, 1, round(self.fs * self.edgelength))
        x[round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength) : 2 * round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength)] = 1
        x[2 * round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength) : 2 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength)] = 1
        x[2 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength) : 3 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength)] = 1
        x[3 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength) : 3 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)] = 1
        x[3 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength) : 4 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)] = 1

        return t, x

    def NAND(self):
        '''Returns a NAND signal with time array t'''
        samples = 4 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)
        x = np.empty(samples)
        t = np.linspace(0, samples/self.fs, samples)

        x[0:round(self.fs * self.signallength / 4)] = 1
        x[round(self.fs * self.signallength / 4) : round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength)] = 1
        x[round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength) : 2 * round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength)] = 1
        x[2 * round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength) : 2 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength)] = 1
        x[2 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength) : 3 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength)] = 1
        x[3 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength) : 3 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)] = np.linspace(1, 0, round(self.fs * self.edgelength))
        x[3 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength) : 4 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)] = 0

        return t, x

    def NOR(self):
        '''Returns a NOR signal with time array t'''
        samples = 4 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)
        x = np.empty(samples)
        t = np.linspace(0, samples/self.fs, samples)

        x[0:round(self.fs * self.signallength / 4)] = 1
        x[round(self.fs * self.signallength / 4) : round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength)] = np.linspace(1, 0, round(self.fs * self.edgelength))
        x[round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength) : 2 * round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength)] = 0
        x[2 * round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength) : 2 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength)] = 0
        x[2 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength) : 3 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength)] = 0
        x[3 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength) : 3 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)] = 0
        x[3 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength) : 4 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)] = 0

        return t, x


    def XNOR(self):
        '''Returns an XNOR signal with time array t'''
        samples = 4 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)
        x = np.empty(samples)
        t = np.linspace(0, samples/self.fs, samples)

        x[0:round(self.fs * self.signallength / 4)] = 1
        x[round(self.fs * self.signallength / 4) : round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength)] = np.linspace(1, 0, round(self.fs * self.edgelength))
        x[round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength) : 2 * round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength)] = 0
        x[2 * round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength) : 2 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength)] = 0
        x[2 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength) : 3 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength)] = 0
        x[3 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength) : 3 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)] = 0
        x[3 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength) : 4 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)] = 1

        return t, x

    #%%
    ####################################################
    ############# INPUT METHODS ########################
    ####################################################
    def BoolInput(self):
        '''Return input signals for Boolean logic (x and y), with a weight array
        W for filtering out the edges. Also returns a time array t.'''
        samples = 4 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)
        x = np.empty(samples)
        y = np.empty(samples)
        W = np.empty(samples)
        t = np.linspace(0, samples/self.fs, samples)

        x[0:round(self.fs * self.signallength / 4)] = 0
        x[round(self.fs * self.signallength / 4) : round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength)] = np.linspace(0, 1, round(self.fs * self.edgelength))
        x[round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength) : 2 * round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength)] = 1
        x[2 * round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength) : 2 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength)] = np.linspace(1, 0, round(self.fs * self.edgelength))
        x[2 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength) : 3 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength)] = 0
        x[3 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength) : 3 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)] = np.linspace(0, 1, round(self.fs * self.edgelength))
        x[3 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength) : 4 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)] = 1

        y[0:round(self.fs * self.signallength / 4)] = 0
        y[round(self.fs * self.signallength / 4) : round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength)] = 0
        y[round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength) : 2 * round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength)] = 0
        y[2 * round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength) : 2 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength)] = np.linspace(0, 1, round(self.fs * self.edgelength))
        y[2 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength) : 3 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength)] = 1
        y[3 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength) : 3 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)] = 1
        y[3 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength) : 4 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)] = 1

        #define weight signal
        W[0:round(self.fs * self.signallength / 4)] = 1
        W[round(self.fs * self.signallength / 4) : round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength)] = 0
        W[round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength) : 2 * round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength)] = 1
        W[2 * round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength) : 2 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength)] = 0
        W[2 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength) : 3 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength)] = 1
        W[3 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength) : 3 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)] = 0
        W[3 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength) : 4 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength)] = 1

        W[round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength): round(self.fs * self.signallength / 4) + round(self.fs * self.edgelength) + 40] = 0
        W[2 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength): 2 * round(self.fs * self.signallength / 4) + 2 * round(self.fs * self.edgelength) + 40] = 0
        W[3 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength): 3 * round(self.fs * self.signallength / 4) + 3 * round(self.fs * self.edgelength) + 40] = 0
        return t, x, y, W

