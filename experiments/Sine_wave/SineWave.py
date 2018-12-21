# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 14:37:08 2018

@author: crazy
"""

import SkyNEt.modules.SaveLib as SaveLib
import matplotlib.pyplot as plt 
from SkyNEt.instruments.ADwin import adwinIO
from SkyNEt.instruments.niDAQ import nidaqIO
import numpy as np 
import os
import config_SiW as config

# Load the information from the config class.
config = config.experiment_config()

#Initialize save directory
saveDirectory = SaveLib.createSaveDirectory(config.filepath, config.name)

# Define the device input using the function in the config class.

Input = config.SineWave( config.Amplitude, config.frequency, config.n_points, config.fs)

#Measure using the device specified in the config class
if config.device == 'nidaq':
	Output = nidaqIO.IO(Input, config.fs)
elif config.device == 'adwin':
	adwin = adwinIO.InitInstrument()
	Output = adwinIO.IO(adwin, Input, config.fs)
else:
	print('specify measurement device')

#save the Input and Output
SaveLib.saveExperiment(saveDirectory, input = Input, output = Output)

# Plot the Square wave
NPoints = np.linspace(0, config.n_points-1, config.n_points)
print(np.shape(Output))
plt.figure()
plt.plot(NPoints, Output)
plt.show()
