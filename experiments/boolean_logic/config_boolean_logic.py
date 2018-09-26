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

        # Define experiment
        self.amplification = 1 #makes up for the different IVVI amplifications, 1G = 1 and 1M = 1000 such that the output is in nA
        self.TargetGen = self.NOR
        self.generations = 1
        self.generange = [[-600,600], [-900, 900], [-900, 900], [-900, 900], [-600, 600], [0.1, 0.5]]

        # Specify either partition or genomes
        #self.partition = [5, 5, 5, 5, 5]
        #self.genomes = 100

        ################################################
        ######### USER-SPECIFIC PARAMETERS #############
        ################################################

        ################# Save settings ################
        self.filepath = r'D:\data\path\to\your\directory'
        self.name = 'AND'

        ############## New Fitness function ############

        ################################################
        ################# OFF-LIMITS ###################
        ################################################
        # Check if genomes parameter has been changed
        if(self.genomes != sum(self.default_partition)):
            if(self.genomes%5 == 0):
                self.partition = [self.genomes%5]*5  # Construct equally partitioned genomes
            else:
                print('WARNING: The specified number of genomes is not divisible by 5.'
                      + ' The remaining genomes are generated randomly each generation. '
                      + ' Specify partition in the config instead of genomes if you do not want this.')
                self.partition = [self.genomes//5]*5  # Construct equally partitioned genomes
                self.partition[-1] += self.genomes%5  # Add remainder to last entry of partition

        self.genomes = sum(self.partition)  # Make sure genomes parameter is correct
        self.genes = len(self.generange)  # Make sure genes parameter is correct

    #####################################################
    ############# USER-SPECIFIC METHODS #################
    #####################################################



        pass
