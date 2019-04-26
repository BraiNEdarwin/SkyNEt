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

    def matrix(self):
        '''
        this function holds the connection matrix.
        It works as follows: every row is connected to 1 BNC output C1..8 resp.
        to this output you can connect 1 pin of a device or 1 pin of multiple devices
        this will then also interconnect them. If you want to interconnect two pins of the same device,
        you have to do it on the PCB.
        the column denotes the device, column 1 is for device 1 etc.
        a 1 corresponds to closed and a 0 to open
        '''
        connect_matrix = np.zeros((8,8))
        connect_matrix[0,:] = [0, 0, 1, 0, 0, 0, 0, 0] # C1 connects to e1 of D(fill in)
        connect_matrix[1,:] = [0, 0, 1, 0, 0, 0, 0, 0] # C2 connects to e2 of D(fill in)
        connect_matrix[2,:] = [0, 0, 1, 0, 0, 0, 0, 0] # C3 connects to e3 of D(fill in)
        connect_matrix[3,:] = [0, 0, 1, 0, 0, 0, 0, 0] # C4 connects to e4 of D(fill in)
        connect_matrix[4,:] = [0, 0, 0, 1, 0, 0, 0, 0] # C5 connects to e5 of D(fill in)
        connect_matrix[5,:] = [0, 0, 0, 1, 0, 0, 0, 0] # C6 connects to e6 of D(fill in)
        connect_matrix[6,:] = [0, 0, 0, 1, 0, 0, 0, 0] # C7 connects to e7 of D(fill in)
        connect_matrix[7,:] = [0, 0, 0, 1, 0, 0, 0, 0] # C8 connects to e8 of D(fill in)
        return(connect_matrix)