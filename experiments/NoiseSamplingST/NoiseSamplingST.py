# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:41:40 2018
This experiment collects data for a specific sampling time. This can be 
used to estimate whether the sampling time is high enough such that the
calculated variance from the currents is reliable.
Furthermore data is collected by measuring a CV config, switching to a random
CV config and the switching back to the set CV config etc. to check whether 
this creates a fluctuation in the  measured currents.

The two different experiments are saved in separate files.

@author: Mark Boon
"""
# SkyNEt imports
import SkyNEt.modules.SaveLib as SaveLib
from SkyNEt.instruments.DAC import IVVIrack
from SkyNEt.instruments.niDAQ import nidaqIO
from SkyNEt.modules.GenericGridConstructor import gridConstructor
import SkyNEt.experiments.NoiseSamplingST.config_NoiseSamplingST as config

# Other imports
import numpy as np
import time
import os

#%% Initialization 

# Initialize config object
config = config.experiment_config()
samples = config.samples
fs = config.fs
T = config.sampleTime

# Initialize save directory
saveDirectoryT = SaveLib.createSaveDirectory(config.filepath, config.name_T)
saveDirectoryS = SaveLib.createSaveDirectory(config.filepath, config.name_S)

# Initialize data container
if config.T_test:
    Tcurrents = np.zeros(samples * config.gridHeight ** len(config.steps), fs * T)
if config.S_test:
    Scurrents = np.zeros(samples * config.gridHeight ** len(config.steps), fs * T)
data = np.zeros((config.T_test + config.S_test) * samples * config.gridHeight ** len(config.steps), fs * T)

# Initialize instruments
ivvi = IVVIrack.initInstrument(dac_step = 500, dac_delay = 0.001)

# Main acquisition loop
controlVoltages = gridConstructor(config.controls, config.steps)

if config.T_test:
    print('Testing accuracy of sample time ...')
    for i in range(0, controlVoltages.shape[0]):
        print('Getting Data for control voltage ' + str(i) + ', ' + str(controlVoltages.shape[0] - i) + ' control voltages remaining.')
        for j in range(samples):
            print('Sampling ' + str(j) + '/' + str(samples) +'...')
            IVVIrack.setControlVoltages(ivvi, controlVoltages[i,:]) 
            time.sleep(2)  # Pause in between two samples
            Tcurrents[i * samples + j,:] = nidaqIO.IO(np.zeros(fs * T + 1), fs) 

if config.S_test:
    print('Testing accuracy of switching CV config ...')
    for i in range(0, controlVoltages.shape[0]):
        print('Getting Data for control voltage ' + str(i) + ', ' + str(controlVoltages.shape[0] - i) + ' control voltages remaining.')
        for j in range(samples):
            print('Sampling ' + str(j) + '/' + str(samples) +'...')
            IVVIrack.setControlVoltages(ivvi, controlVoltages[i,:]) 
            time.sleep(2)  # Pause in between two samples
            Scurrents[i * samples + j,:] = nidaqIO.IO(np.zeros(fs * T + 1), fs) 
            IVVIrack.setControlVoltages(ivvi, np.random.random(7) * 1400 - 700) # Switch to a random config.
            time.sleep(1) # Keep the CV config on random for a short period

# Set IVVI back to zero
IVVIrack.setControlVoltages(ivvi, np.zeros(8))   


# Save obtained data (the two tests are saved in separate files)
if config.T_test:
    np.savez(os.path.join(saveDirectoryT, 'nparrays'), CV = controlVoltages, output = Tcurrents)

if config.S_test:
    np.savez(os.path.join(saveDirectoryS, 'nparrays'), CV = controlVoltages, output = Scurrents)
    


