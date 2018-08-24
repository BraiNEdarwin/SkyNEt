#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 18:22:23 2018

@author: hruiz
"""

import numpy as np
from config.config_class import config_class

class experiment_config(config_class):
    '''This is a template for the configuration class that is experiment/user specific.
    It inherits from config_class default values that are known to work well with boolean logic.
    You can define user-specific parameters in the construction of the object in __init__() or define 
    methods that you might need after, e.g. a new fitness function or input and output generators. 
    Remember if you define a new fitness function or generator, you have to redefine the self.Fitness,
    self.Target_gen and self.Input_gen in __init__()
    '''
    
    def __init__(self):
        super().__init__() #DO NOT REMOVE!
        
        ################# Experiment ###################
        self.Target_gen = self.XNOR

        ################# Save settings ################
        self.filepath = r'some_path/to/save'
        self.name = 'AND'

        ################################################
        ############### Evolution settings #############
        ################################################
        self.genes = 6
        self.genomes = 25
        self.generations = 500
        self.generange = [[-900,900], [-900, 900], [-900, 900], [-900, 900], [-900, 900], [0.1, 2]]
        self.genelabels = ['CV1/T11','CV2/T13','CV3/T17','CV4/T7','CV5/T1', 'Input scaling']
        self.fitnessparameters = [1, 0, 1, 0.01]    

        #control signallength and measurement speed.
        self.signallength = 0.5  #in seconds 
        self.edgelength = 0.01  #in seconds
        self.fs = 1000
        ################################################
        ######### USER-SPECIFIC PARAMETERS #############
        ################################################
        

        
        ############## New Fitness function ############
        #self.Fitness = self.NewFitness
    
    #####################################################    
    ############# USER-SPECIFIC METHODS #################
    #####################################################
    def NewFitness(self):
        '''Here define the new fitness for your experiment if it is not in the parent class 
        Remember to comment!!
        '''
        pass
    
    