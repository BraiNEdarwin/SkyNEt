import numpy as np 

class experiment_config(object):

    def __init__(self):



        #define the constants
        self.v_low = -25
        self.v_high = 45 
        self.frequency = 5
        self.n_points = 1000
        self.position = 'high' 

    def SquareWave(self, v_high, v_low, n_points):
        Input = [0]*n_points
        n_points = int(n_points/4)
        slope = 100
        Input_L = np.linspace(v_low, v_low, n_points-slope)
        Input_LH = np.linspace(v_low, v_high, slope)
        Input_H = np.linspace(v_high, v_high, n_points-slope)
        Input_HL = np.linspace(v_high, v_low, slope)
        Input[0:n_points-slope] = Input_L
        Input[n_points-slope:n_points] = Input_LH
        Input[n_points:2*n_points-slope] = Input_H
        Input[2*n_points-slope:2*n_points] = Input_HL 
        Input[2*n_points:3*n_points-slope] = Input_L
        Input[3*n_points-slope:3*n_points] = Input_LH 
        Input[3*n_points:4*n_points-slope] = Input_H
        Input[4*n_points-slope:4*n_points] = Input_HL

        return Input