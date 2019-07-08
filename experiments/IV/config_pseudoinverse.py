import numpy as np
import os
import matplotlib.pyplot as plt
import math

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
        self.filepath = '/home/Darwin/data/Rik/pseudoinverse/'
        self.name = 'test'
        self.configSrc = os.path.dirname(os.path.abspath(__file__))
        
        #define two input voltages and the amount of points you want to measure
        self.numberinputs =2
        self.vc =7
        self.vpoints=  [[-0.5,-0.6],
                       [0.5,-0.6],
                       [0,-0.8],
                       [0, 0.8],
                       [-0.5,0.6],
                       [0.5,0.6],
                       [-0.8,0.1],
                       
                       [0,0]]#to end the measurement on 0V
        self.points_per_input=150
        self.n_points =self.points_per_input*self.vc
        self.slopesize= 50

        #define the input and output amplifications.
        self.amplificatin = 1
        self.source_gain = 1

        #measurment tool settings.
        self.device = 'nidaq'
        self.fs = 1000
        

        



    def Sweepgen(self, v_high, v_low, n_points,backgate, direction):
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
        
        Input = np.ones((1,len(Input1)+len(Input2)+len(Input3)))*backgate
        Input[0, 0:len(Input1)] = Input1
        Input[0, len(Input1):len(Input1)+len(Input2)] = Input2
        Input[0, len(Input1)+len(Input2):len(Input1)+len(Input2)+len(Input3)] = Input3

        return Input
    
    def Pulse(self, v_high, v_low, n_points, n_pulses):
        
        Input = np.zeros((1,n_points))
        pulse_length = n_points/(n_pulses)
        for i in range(n_pulses):
            Input[0,i*pulse_length:(i+1)*pulse_length/2] = np.ones(pulse_length/2)*v_low
            Input[0,(i+1)*pulse_length/2:(i+1)*pulse_length] = np.ones(pulse_length/2)*v_high
        
        return Input
    
    def sine(self, n_points, frequency, amplitude):
        
        
        points = np.linspace(0,1,n_points-1)
        Input = amplitude*np.sin(points* np.pi / 180.)
        
        
        return Input
        
    def pseudoInput(self, vpoints, vc, numberinputs, points_per_input, slopesize):
        noslope=points_per_input-slopesize
        Input = np.zeros((numberinputs,points_per_input*vc))
        print(Input.shape)
        for num in range(numberinputs):
            for i in range(vc):              
                Input[num,i*points_per_input:i*points_per_input+noslope]=np.ones(noslope)*vpoints[i][num]
                Input[num,i*points_per_input+noslope:(i+1)*points_per_input]=np.linspace(vpoints[i][num], vpoints[i+1][num],slopesize)
        return Input
    


# Read config.txt file
#exec(open("config_software.txt").read())

#loads .npz file
    def generate_cp(n=100, mean_I0=-0.3, mean_I1=-0.3, amp_I0=0.9, amp_I1=0.9):
         values_I0 = [mean_I0-amp_I0+amp_I0*2/2*(i//n//7) for i in range(21*n)]
         values_I1 = [mean_I1-amp_I1+amp_I1*2/6*(i//n%7) for i in range(21*n)]
         input_data = np.array([[values_I0],[values_I1]])
         targets = np.array([0,0,0,1,1,1,1,0,1,1,1,1,2,2,1,1,2,1,2,1,2])
         target_data = np.zeros((2100))
         for i in range(len(targets)):
             target_data[i*100:i*100+100] = np.ones(100)*targets[i]
         return input_data, target_data








