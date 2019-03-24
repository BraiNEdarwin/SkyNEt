'''
This script applies control+input voltage configurations defined in the
config file with the IVVI dacs.
The output is measured through the IVVI current amplifier.
All 7 dacs are passed through a 1kOhm resistor and the voltage drop over 
them is measured with the help of a resistor box. 
'''
# SkyNEt imports
import SkyNEt.modules.SaveLib as SaveLib
import config_measure_eight_electrode as config
from SkyNEt.instruments import InstrumentImporter
from SkyNEt.instruments.Keithley2400 import Keithley2400

# Other imports
import time
import numpy as np
import matplotlib.pyplot as plt

# Initialize config object
cf = config.experiment_config()

# Initialize data array
data = np.zeros((cf.control_sequence.shape[0], 8, 4, 103))

# Initialize instruments
ivvi = InstrumentImporter.IVVIrack.initInstrument()

P = [0, 1, 0, 1]
Q = [0, 0, 1, 1]
for ii in range(cf.control_sequence.shape[0]):
    print(f'Now measuring control sequence {ii}')
    controlVoltages = cf.control_sequence[ii].copy()
    for jj in range(4):
        # Prepare controlVoltages
        controlVoltages[0] = P[jj]*cf.control_sequence[ii, 0]
        controlVoltages[1] = Q[jj]*cf.control_sequence[ii, 1]

        # Set the DAC voltages
        InstrumentImporter.IVVIrack.setControlVoltages(ivvi, controlVoltages)
        
        time.sleep(1)  # Wait after setting DACs

        # Measure output with nidaq
        currents = InstrumentImporter.nidaqIO.IO(np.zeros(cf.N), cf.fs, inputPorts = [1, 1, 1, 1, 1, 1, 1, 1]) 
        
        plt.figure()
        for kk in range(currents.shape[0]):
            plt.plot(currents[kk], label = {kk})
        plt.legend()
        
        # Convert voltages to currents
        for kk in range(7):
            currents[kk] = (controlVoltages[kk]/1E3 - currents[kk])/cf.resistance
           
        # Convert output current to A
        currents[-1] = currents[-1]*cf.amplification*1E-9

        # Store result in input_output
        mean = np.mean(currents, axis = 1)
        std = np.std(currents, axis = 1)
        data[ii, :, jj, 0] = controlVoltages
        data[ii, :, jj, 1] = mean
        data[ii, :, jj, 2] = std
        data[ii, :, jj, 3:] = currents
        

# Save experiment
saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)
SaveLib.saveExperiment(saveDirectory,
                       data = data,
                       )

print(data[0, :, :, 1])

InstrumentImporter.reset(0, 0)

