import SkyNEt.modules.SaveLib as SaveLib
import matplotlib.pyplot as plt 
from SkyNEt.instruments.ADwin import adwinIO
from SkyNEt.instruments.niDAQ import nnidaqIO
import numpy as np 
import os
import config_SW as config

# Load the information from the config class.
config = config.experiment_config()

#Initialize save directory
saveDirectory = SaveLib.createSaveDirectory(config.filepath, config.name)

# Define the device input using the function in the config class.
Input = config.SquareWave( config.v_high, config.v_low, config.n_points)

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
plt.figure()
plt.plot(n_points[0:len(Output)], Output)
plt.show()