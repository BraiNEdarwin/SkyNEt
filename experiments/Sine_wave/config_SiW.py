# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 14:21:02 2018

@author: crazy
"""

import matplotlib.pyplot as py
import numpy as np

class experiment_config(object):
    
    def __init__(self):


        #define where you want to save the data.
        self.filepath = r'F:\test'
        self.name = 'test'
        
        #define the Parameters.
        self.Amplitude = 0.1
        self.n_points = 1000
        self.frequency = 25

        #define the input and output amplifications.
        self.amplification = 1
        self.source_gain = 1

        #measurment tool settings.
        self.device = 'nidaq'
        self.fs = int(self.frequency*self.n_points)


        self.SineWave = self.SineWave
        
    def SineWave(self, Amplitude, frequency, n_points, fs):
        counter = 0
        Input= [0]*n_points
        ln_points = np.linspace(0, n_points-1, n_points)
        for i in range (0, n_points):
            Input[counter] = Amplitude*np.sin(2*np.pi*frequency*ln_points[counter]/fs)
            counter = counter + 1

        return Input
