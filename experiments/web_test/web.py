""" Copy of IV.py """

#import SkyNEt.modules.SaveLib as SaveLib
import SkyNEt.modules.Evolution as Evolution
import matplotlib.pyplot as plt
from SkyNEt.instruments import InstrumentImporter
from SkyNEt.instruments.Keithley2400 import Keithley2400
import numpy as np
import os
import config_web as config

# Load the information from the config class.
cf = config.experiment_config()

# XOR gate
gene = np.array([ 0.65239738,  0.58900355,  0.76227318,  0.51687633,  0.40839594,  0.58702374])
#gene = np.array([ 0.64679704,  0.58900355,  0.76227318,  0.51687633,  0.40939317, 0.58428664])
#gene = np.array([ 0.23442611,  0.44156965,  0.74649462,  0.55534413,  0.41519953, 0.5878727 ])
generange = [[-900,900], [-900, 900], [-500, 500], [-500, 500], [-900, 900], [0.1, 0.9]]
input_data = cf.voltage_from_result(gene, generange)

nr_samples = 10

powergrid_frequency = 50

# Initialize save directory.
#saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)


# arduino switch network
# Initialize serial object
ser = InstrumentImporter.switch_utils.init_serial(cf.switch_comport)
# Switch to device
InstrumentImporter.switch_utils.connect_single_device(ser, cf.switch_device)
# Status print
print('INFO: Connected device %i' % cf.switch_device)

# Measure using the device specified in the config class.
if cf.measure_device == 'keithley2400':
    keithley = Keithley2400.Keithley_2400('keithley', 'GPIB0::11')
    keithley.compliancei.set(1E-6)
    keithley.compliancev.set(4)
#    keithley.nplci(0.01)
    keithley.output.set(1)
else:
    raise('specify measurement device')

# set voltages with
if cf.set_device == 'cdaq':
    sdev = InstrumentImporter.nidaqIO.IO_cDAQ(nr_channels=7)
else:
    raise('specify set device')

output = np.zeros(4*nr_samples)
for ii, single_input in enumerate(input_data):
    sdev.ramp(single_input)
    for jj in range(nr_samples):
        output[ii*nr_samples+jj] = keithley.curr()

sdev.ramp_zero()


if cf.measure_device == 'keithley2400':
    keithley.output.set(0)
    keithley.close()

InstrumentImporter.switch_utils.close(ser)

plt.figure()
plt.plot(output*1e9, '-o')

# Save the Input and Output
#SaveLib.saveExperiment(saveDirectory, input = input_data, output = output, gene=gene)


## Plot the IV curve.
#plt.figure()
#plt.plot(Input[0:len(Output)], Output)
#plt.show()
#
## Final reset
#InstrumentImporter.reset(0, 0)
