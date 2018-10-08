# -*- coding: utf-8 -*-
"""
Script to measure noise using a grid of control voltages

@author: Mark Boon (Oct 2018)
"""
# SkyNEt imports
from SkyNEt.instruments.DAC import IVVIrack
from SkyNEt.instruments.niDAQ import nidaqIO
from SkyNEt.modules.GenericGridConstructor import gridConstructor
import SkyNEt.modules.SaveLib as SaveLib
import SkyNEt.config_NoiseGrid as config

# Other imports
import numpy as np
import os
import time

#%% Initialization 

# Initialize config object
config = config.experiment_config()
fs = config.fs
T = config.sampleTime

# Initialize data container
currents = np.zeros(config.gridHeight ** len(config.steps), fs * T)

# Initialize save directory
saveDirectory = SaveLib.createSaveDirectory(config.filepath, config.name)

# Initialize instruments
ivvi = IVVIrack.initInstrument(dac_step = 500, dac_delay = 0.001)

# Main acquisition loop
controlVoltages = gridConstructor(config.controls, config.steps)
for i in range(0, controlVoltages.shape[0]):
    print('Getting Data for control voltage ' + str(i) + ', ' + str(controlVoltages.shape[0] - i) + ' control voltages remaining.')
    IVVIrack.setControlVoltages(ivvi, controlVoltages[i,:]) 
    time.sleep(0.5)  # Wait after setting DACs # TODO: change wait time with grid step size
    currents[i,:] = nidaqIO.IO(np.zeros(fs * T + 1), fs) 

# Set IVVI back to zero
IVVIrack.setControlVoltages(ivvi, np.zeros(8))   

 
# Saving the data:
np.savez(os.path.join(saveDirectory, 'nparrays'), CV = controlVoltages, output = currents)


    
