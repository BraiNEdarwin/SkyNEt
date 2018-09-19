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

        #define experiment
        self.amplification = 1 #makes up for the different IVVI amplifications, 1G = 1 and 1M = 1000 such that the output is in nA
        self.TargetGen = self.NOR
        self.partition = [2, 2, 2, 2, 2]
        self.generations = 1
        self.genomes = 1000
        self.fs = 500
        self.signallength = 0.45
        self.edgelength = 0.01
        self.generange = [[-600,900], [-1200, 1200], [-1200, 1200], [-1200, 1200], [-600, 600], [0.1, 0.5]]
        ################################################
        ######### USER-SPECIFIC PARAMETERS #############
        ################################################

        ################# Save settings ################
        self.filepath = r'D:\Tao\test'
        self.name = 'test'

        ############## New Fitness function ############

        ################################################
        ################# OFF-LIMITS ###################
        ################################################
        self.genomes = sum(self.partition)  # Make sure genomes parameter is correct
        self.genes = len(self.generange)  # Make sure genes parameter is correct

    #####################################################
    ############# USER-SPECIFIC METHODS #################
    #####################################################



        pass
