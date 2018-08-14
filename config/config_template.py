#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 18:22:23 2018

@author: hruiz
"""

import numpy as np
from dev_v1.config_class import config_class

#TODO: change dev_v1 to the correct package name when finished

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
        
        ################################################
        ######### USER-SPECIFIC PARAMETERS #############
        ################################################
        
        ################# Save settings ################
        self.filepath = r'some_path/to/save'
        self.name = 'AND'
        
        ############## New Fitness function ############
        self.Fitness = self.new_fitness
    
    #####################################################    
    ############# USER-SPECIFIC METHODS #################
    #####################################################
    def new_fitness(self):
        '''Here define the new fitness for your experiment if it is not in the parent class 
        Remember to comment!!
        '''
        pass
    
    