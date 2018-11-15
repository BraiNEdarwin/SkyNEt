#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config for testing the Evolution module.
"""

import numpy as np
from SkyNEt.config.config_class import config_class

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
        self.generange = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
        self.partition = [5, 5, 5, 5, 5]

        ################# Save settings ################
        self.filepath = 'some_path/to/save'
        self.name = 'AND'

        ############## New Fitness function ############
        self.Fitness = self.Convex

    #####################################################
    ############# USER-SPECIFIC METHODS #################
    #####################################################
    def Convex(self, y):
        '''A simple convex fitness function that will attain a maximum value of
        1 for y = 0. Assumes that y is a column vector!'''
        return 1/(1 + np.dot(y.T, y))
