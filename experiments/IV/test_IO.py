import SkyNEt.modules.SaveLib as SaveLib
import matplotlib.pyplot as plt
from SkyNEt.instruments.ADwin import adwinIO
from SkyNEt.instruments.niDAQ import nidaqIO
import numpy as np
import os
import config_IV as config

# Load the information from the config class.
config = config.experiment_config()

# Initialize save directory.
saveDirectory = SaveLib.createSaveDirectory(config.filepath, config.name)

# Define the device input using the function in the config class.
Input = np.ones((2,config.n_points))
Input[1] = config.Sweepgen( config.v_high, config.v_low, config.n_points, config.direction)

# Measure using the device specified in the config class.
if config.device == 'nidaq':
    Output = nidaqIO.IO_all(Input, config.fs, 2, 8)
elif config.device == 'adwin':
    adwin = adwinIO.InitInstrument()
    Output = adwinIO.IO(adwin, Input, config.fs)
else:
    print('specify measurement device')

# Save the Input and Output
SaveLib.saveExperiment(saveDirectory, input = Input, output = Output)

# Plot the IV curve.
NPoints = np.linspace(0, config.n_points-1, config.n_points)
print(np.shape(Output))
plt.figure()
plt.plot(NPoints, Output[0])
plt.figure()
plt.plot(NPoints, Output[1])
plt.figure()
plt.plot(NPoints, Output[2])
plt.show()