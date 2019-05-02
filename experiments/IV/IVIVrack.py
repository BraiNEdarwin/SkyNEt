import SkyNEt.modules.SaveLib as SaveLib
import matplotlib.pyplot as plt
from SkyNEt.instruments import InstrumentImporter
from SkyNEt.instruments.Keithley2400 import Keithley2400
import numpy as np
import os
import config_IV as config

# Load the information from the config class.
config = config.experiment_config()

# Initialize save directory.
saveDirectory = SaveLib.createSaveDirectory(config.filepath, config.name)

# Define the device input using the function in the config class.
Input = config.Sweepgen( config.v_high, config.v_low, config.n_points, config.direction)
if config.device2 == 'IVVI':
	ivvi = InstrumentImporter.IVVIrack.initInstrument()
#	InstrumentImporter.IVVIrack.setControlVoltages(ivvi, config.controlVoltages)

# Measure using the device specified in the config class.
if config.device == 'nidaq':
    Output = InstrumentImporter.nidaqIO.IO(Input, config.fs)
elif config.device == 'adwin':
    adwin = InstrumentImporter.adwinIO.initInstrument()
    Output = InstrumentImporter.adwinIO.IO(adwin, Input, config.fs, inputPorts = [1, 1, 1, 1, 1, 1, 1])
elif config.device == 'keithley':
    Output = np.zeros_like(Input)
    keithley = Keithley2400.Keithley_2400('keithley', 'GPIB0::11')


    # Set compliances
    keithley.compliancei.set(1E-6)
    keithley.compliancev.set(4)

    # Turn keithley output on
    keithley.output.set(1)

    for ii in range(7):
        InstrumentImporter.IVVIrack.setControlVoltage(ivvi, config.controlVoltages[ii],0)

        # Record current
        #time.sleep(0.05)
        Output[ii] = keithley.curr()
        print(Output)

    # Turn keithley output off
    keithley.output.set(0)
else:
    print('specify measurement device')

# Save the Input and Output
SaveLib.saveExperiment(saveDirectory, input = Input, output = Output)

x = np.linspace(0,1,len(Output))
# Plot the IV curve.
for i in range(1):
    plt.figure()
    plt.plot(x, Output)
plt.show()

# Final reset
InstrumentImporter.reset(0, 0)
