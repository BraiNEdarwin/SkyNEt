# -*- coding: utf-8 -*-

class experiment_config():
    
    
    def __init__(self):
        
        self.fs = 1000          # Sampling freq (Hz)
        self.sampleTime = 2     # Sampling time (s)
        self.amplification = 100E6
        self.steps = [-600, -300, 0, 300, 600]    # Steps per control voltage
        self.gridHeight = len(self.steps[0])        # Amount of steps for a CV
        self.controls = 7
        
        
        self.filepath = 'D:\data\Mark\noise'
        self.name = 'test'
        
        
        
    