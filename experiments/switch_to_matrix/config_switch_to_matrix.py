from SkyNEt.config.config_class import config_class
import numpy as np


class experiment_config(config_class):
    '''
    This is the config file for switching the switch network setup
    to a particular device. There are actually only two parameters:

    comport; string, e.g. 'COM3' the comport to which the arduino is 
                connected
    matrix; string, an 8x8 matrix with connection configurations for the bnc
    '''

    def __init__(self):
        super().__init__() #DO NOT REMOVE!

        ################################################
        ######### SPECIFY PARAMETERS ###################
        ################################################
        self.comport = 'COM3'  # COM port of the arduino
        self.matrix = self.matrix

    def matrix():
        matrix = np.zeros((8,8))
        matrix[0,:] = [0, 0, 1, 0, 0, 0, 0, 0] # C1 connects to e1 of D(fill in)
        matrix[1,:] = [0, 0, 1, 0, 0, 0, 0, 0] # C2 connects to e2 of D(fill in)
        matrix[2,:] = [0, 0, 1, 0, 0, 0, 0, 0] # C3 connects to e3 of D(fill in)
        matrix[3,:] = [0, 0, 1, 0, 0, 0, 0, 0] # C4 connects to e4 of D(fill in)
        matrix[4,:] = [0, 0, 0, 1, 0, 0, 0, 0] # C5 connects to e5 of D(fill in)
        matrix[5,:] = [0, 0, 0, 1, 0, 0, 0, 0] # C6 connects to e6 of D(fill in)
        matrix[6,:] = [0, 0, 0, 1, 0, 0, 0, 0] # C7 connects to e7 of D(fill in)
        matrix[7,:] = [0, 0, 0, 1, 0, 0, 0, 0] # C8 connects to e8 of D(fill in)
        return(matrix)