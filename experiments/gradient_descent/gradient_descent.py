# -*- coding: utf-8 -*-

"""
Created on Sun Jan 27 12:29:46 2019

This script applies gradient descent directly on the device.
For now it is only possible to find boolean logic with this experiment

@author: Mark Boon
"""
import SkyNEt.experiments.gradient_descent.config_gradient_descent as config
import SkyNEt.modules.SaveLib as SaveLib
from SkyNEt.instruments import InstrumentImporter
from SkyNEt.instruments.DAC import IVVIrack
import SkyNEt.modules.PlotBuilder as PlotBuilder

import time
import numpy as np
import random

# Initialize config object
cf = config.experiment_config()

# Initialize input and target
t = cf.InputGen()[0]  # Time array
x = np.asarray(cf.InputGen()[1:3])  # Array with P and Q signal
w = cf.InputGen()[3]  # Weight array
target = cf.gainFactor * cf.targetGen()[1]  # Target signal

# Initialize arrays
controls = np.zeros((cf.n, cf.controls))
controls[0,:] = (np.random.random(cf.controls) - 0.5) * 2 * cf.Vmax # First (random) controls
data = np.zeros((cf.n, x.shape[1]))
waves = np.zeros((cf.controls, x.shape[1]))
for i in range(cf.controls):
    waves[i,:] = np.sin(2*np.pi*t*cf.freq[i]) * cf.waveAmplitude


# Main aqcuisition loop

for i in range(cf.n):
    # Maybe increase the time for the first input for x a bit to avoid transients (since the controls always come from 0V)



    # Fourier transform on output data to find dI/dV
