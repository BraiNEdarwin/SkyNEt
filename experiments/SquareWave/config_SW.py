import numpy as np 

class experiment_config(object):

    def __init__(self):

        #define where you want to save the data.
        self.filepath = r'C:\test'
        self.name = 'test'

        #define the constants
        self.v_low = 0
        self.v_high = 1 
        self.frequency = 25
        self.n_points = 10000
        self.position = 'high'

        #define the input and output amplification
        self.amplification = 1
        self.source_gain = 1

        #measurement tool settings.
        self.device = 'nidaq'
        self.fs = self.frequency * self.n_points

    def SquareWave(self, v_high, v_low, n_points):
        Input = [0]*n_points
        n_points = int(n_points/4)

        Input_L = np.linspace(v_low, v_low, n_points)
        Input_H = np.linspace(v_high, v_high, n_points)

        Input[0:n_points] = Input_L
        Input[n_points:2*n_points] = Input_H
        Input[2*n_points:3*n_points] = Input_L
        Input[3*n_points:4*n_points] = Input_H

        return Input