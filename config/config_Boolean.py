
import numpy as np
from config_class import config_class

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
        self.amplification = 1 #makes up for the different IVVI amplifications, 1G = 1 and 1M = 1000
        self.TargetGen = self.NOR
        ################################################
        ######### USER-SPECIFIC PARAMETERS #############
        ################################################
        
        ################# Save settings ################
        self.filepath = r'some_path/to/save'
        self.name = 'AND'
        
        ############## New Fitness function ############

    
    #####################################################    
    ############# USER-SPECIFIC METHODS #################
    #####################################################



        pass
    