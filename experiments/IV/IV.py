import SkyNEt.modules.SaveLib as SaveLib
import matplotlib.pyplot as plt
from SkyNEt.instruments import InstrumentImporter
import numpy as np
import os
import config_IV as config

# Load the information from the config class.
config = config.experiment_config()

# Initialize save directory.
saveDirectory = SaveLib.createSaveDirectory(config.filepath, config.name)

# Define the device input using the function in the config class.
Input = config.Sweepgen( config.v_high, config.v_low, config.n_points, config.direction)

# Measure using the device specified in the config class.
if config.device == 'nidaq':
    Output = InstrumentImporter.nidaqIO.IO(Input, config.fs)
elif config.device == 'adwin':
    adwin = InstrumentImporter.adwinIO.initInstrument()
    Output = InstrumentImporter.adwinIO.IO(adwin, Input, config.fs)
else:
    print('specify measurement device')

# Save the Input and Output
SaveLib.saveExperiment(saveDirectory, input = Input, output = Output*config.amplification)

# Plot the IV curve.
plt.figure()
plt.plot(Input[0], Output[0])
plt.show()

# Final reset
InstrumentImporter.reset(0, 0)
