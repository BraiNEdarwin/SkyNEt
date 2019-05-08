""" copy of config_IV.py """


import numpy as np

class experiment_config(object):
    '''This is the configuration file used for IV measurements. 
    The script is designed such that the voltage needs to start and end at zero. This results in v_low needing to be zero or lower and v_high zero or higher.

    Parameter and method list
    filepath; Defines where you want to create a folder in which the output and config files are stored.
    bane; Defines the name of the folder in which the store the output and config files.
    v_low; Controls the lowest voltage in the measurement, needs to be zero or lower.
    v_high; Controls the highest voltage in the measurement, needs to be zero or higher.
    n_points; This is used to define the number of points in the V sweep and therefore defines a part of the sweeprate.
    direction; Controls the sweep direction. down goes to v_low first while up goes to v_high first.
    amplification; This is used to correct for the amplication used in the amplifier (1G=1, 100M=10, 10M=100, 1M=1000).
    source_gain; Defines the source gain. THe input from the nidaq is limited by the IVVI rack. an amplifier can be used to get 5X higher voltages. to correct for this the source_gain needs to be set to 5.
    device; Here you indicate which meanrement device you use (nidaq or adwin).
    fs; Controls the samepling rate of the measurement device.
    sweepgen; This is the function used to generate the input sequence.
    '''

    def __init__(self):


        #define where you want to save the data.
        self.filepath = r'D:\Lennart\tests\\'
        self.name = 'test1'
        
        #define the IV you want to take in volts.
        self.v_low = -0.9
        self.v_high = 0.9
        self.n_points = 10000
        self.direction = 'up'

        #define the input and output amplifications.
        self.amplification = 1
        self.source_gain = 1

        #measurment tool settings.
        self.device = 'nidaq'
        self.fs = 1000


        self.Sweepgen = self.Sweepgen

    def Sweepgen(self, v_high, v_low, n_points, direction):
        n_points = n_points/2

        if direction == 'down':
            Input1 = np.linspace(0, v_low, int((n_points*v_low)/(v_low-v_high)))
            Input2 = np.linspace(v_low, v_high, n_points)
            Input3 = np.linspace(v_high, 0, int((n_points*v_high)/(v_high-v_low)))
        elif direction == 'up':
            Input1 = np.linspace(0, v_high, int((n_points*v_high)/(v_high-v_low)))
            Input2 = np.linspace(v_high, v_low, n_points)
            Input3 = np.linspace(v_low, 0, int((n_points*v_low)/(v_low-v_high)))
        else:
            print('Specify the sweep direction')


        Input = np.zeros(len(Input1)+len(Input2)+len(Input3))
        Input[0:len(Input1)] = Input1
        Input[len(Input1):len(Input1)+len(Input2)] = Input2
        Input[len(Input1)+len(Input2):len(Input1)+len(Input2)+len(Input3)] = Input3
        return Input






