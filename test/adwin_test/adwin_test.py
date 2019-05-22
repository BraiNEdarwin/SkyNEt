from SkyNEt.instruments import InstrumentImporter
import matplotlib.pyplot as plt
import numpy as np
import os
import config_adwin_test as config

# Load the information from the config class.
cf = config.experiment_config()

# Load input
t, x = cf.Generate_input() 

adwin = InstrumentImporter.adwinIO.initInstrument()
y = InstrumentImporter.adwinIO.IO(adwin, 
                                  x, 
                                  cf.fs, 
                                  inputPorts=[1, 1, 1, 1, 0, 0, 0, 0])

# Detect if output is the same as input
if(np.max(abs(x-y)) < 0.001):
    print('ADwin works correctly')

# Final reset
InstrumentImporter.reset(0, 0, exit=False)

# Plot the output curve.
for i in range(4):
    plt.figure()
    plt.plot(t, x[i], 'r--', label = 'Analog output signal')
    plt.plot(t, y[i], 'b', label = 'Analog input signal')
    plt.title(f'AO/AI {i+1}')
    plt.xlabel('t (s)')
    plt.ylabel('V')
    plt.legend()
plt.show()

