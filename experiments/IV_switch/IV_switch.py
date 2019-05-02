import SkyNEt.modules.SaveLib as SaveLib
import matplotlib.pyplot as plt
from SkyNEt.instruments import InstrumentImporter
from SkyNEt.instruments.Keithley2400 import Keithley2400
import numpy as np
import os
import config_IV_switch as config
import time

# Load the information from the config class.
config = config.experiment_config()

# Initialize save directory.
saveDirectory = SaveLib.createSaveDirectory(config.filepath, config.name)

# Define the device input using the function in the config class.
Input = config.Sweepgen( config.v_high, config.v_low, config.n_points, config.direction)
grounded_input = np.zeros((4,Input.shape[0]))
grounded_input[0] = Input
grounded_input[2] = -Input
# Measure using the device specified in the config class.
if config.device == 'nidaq':
    Output = InstrumentImporter.nidaqIO.IO(Input, config.fs)
elif config.device == 'adwin':
    adwin = InstrumentImporter.adwinIO.initInstrument()
    Output = InstrumentImporter.adwinIO.IO(adwin, grounded_input, config.fs, inputPorts = [1, 1, 1, 1, 1, 1, 1])


# elif config.device == 'keithley':
#     Output = np.zeros_like(Input)
#     keithley = Keithley2400.Keithley_2400('keithley', 'GPIB0::11')

#     # Set compliances
#     keithley.compliancei.set(1E-6)
#     keithley.compliancev.set(4)

#     # Turn keithley output on
#     keithley.output.set(1)

#     for ii in range(len(Input)):
#         # Set voltage
#         keithley.volt.set(Input[ii])

#         # Record current
#         time.sleep(0.05)
#         Output[ii] = keithley.curr()

#     # Turn keithley output off
#     keithley.output.set(0)
# else:
#     print('specify measurement device')

# Save the Input and Output
SaveLib.saveExperiment(saveDirectory, input = Input, output = Output*config.amplification)

# Convert to current
#R = 1E6  # Ohm
#V = grounded_input[0] - Output[0]
#I = Output[0]/R
# Plot the IV curve
for n in range(7):
    plt.figure()
    plt.plot(grounded_input[0], Output[n],'b',grounded_input[1], Output[n],'g',grounded_input[2], Output[n],'r',grounded_input[3], Output[n],'k')
# plt.figure()
# plt.plot(grounded_input[0], Output[0],'b',grounded_input[0], Output[1],'g',grounded_input[0], Output[2],'k',grounded_input[0], Output[3],'r')
plt.show()
# Final reset
InstrumentImporter.reset(0, 0)



