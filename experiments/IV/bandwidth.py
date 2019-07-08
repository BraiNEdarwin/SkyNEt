import SkyNEt.modules.SaveLib as SaveLib
import matplotlib.pyplot as plt
from SkyNEt.instruments import InstrumentImporter
import numpy as np
import os
import config_bandwidth as config

# Load the information from the config class.
config = config.experiment_config()
fraction = np.zeros(len(config.frequencysweep))
# Initialize save directory.
saveDirectory = SaveLib.createSaveDirectory(config.filepath, config.name)
for i in range(len(config.frequencysweep)):

# Define the device input using the function in the config class.
# Input = np.zeros([2, config.n_points])
    sine = config.sine( config.n_points, config.frequencysweep[i], config.amplitude)
    print(np.shape(sine))
#time=np.linspace(0,1,1000)
##Input = config.Pulse(config.v_high, config.v_low, config.n_points, config.n_pulses)
#Input = config.sine(config.n_points, 1, config.v_high)

# Measure using the device specified in the config class.
    if config.device == 'nidaq':
        Output = InstrumentImporter.nidaqIO.IO_cDAQ9132(sine, config.fs)
    elif config.device == 'adwin':
        adwin = InstrumentImporter.adwinIO.initInstrument()
        Output = InstrumentImporter.adwinIO.IO(adwin, sine, config.fs)
    else:
        print('specify measurement device')

# Save the Input and Output
#SaveLib.saveExperiment(config.configSrc, saveDirectory, input = Input, output = Output)
#print(Input)
#print(Output)
# Final reset
#InstrumentImporter.reset(0, 0)
    print(np.shape(Output))
    time=np.linspace(0,config.n_points/config.fs, config.n_points)

    fraction[i] = np.max(Output[0,100:])/np.max(sine[0,100:])
    print(fraction)

# Plot the IV curve. only 5 times
    if i%(int(len(config.frequencysweep)/5)) ==0 :
        plt.figure()
        plt.title('frequency = %i' %config.frequencysweep[i])
        plt.plot(time, Output[0],'r')
        plt.plot(time,sine[0],'b')

plt.figure()
plt.title('Bandwidth of resistor measurement')
plt.xlabel('frequency (Hz)')
plt.ylabel('output gain')
plt.plot(config.frequencysweep, fraction,'-o')
plt.show()

# Final reset
InstrumentImporter.reset(0, 0)
# Since InstrumentImporter is not working properly, use adwin reset directly: OLD?

