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
        self.n_points = 10000
        self.frequency = 100

        #define the input and output amplifications.
        self.amplification = 1
        self.source_gain = 1

        #measurment tool settings.
        self.device = 'nidaq'
        self.fs = 1000


        self.SineWave = self.Sinewave
        
    def SineWave(self, Amplitude, frequency, n_points):
        counter = 0
        for i in range (0, n_points):
            Input[counter] = np.sin(2*np.pi*frequency*n_points/fs)
            counter = counter + 1
