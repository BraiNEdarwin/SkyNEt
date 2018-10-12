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
import modules.SaveLib as SaveLib
from instruments.DAC import IVVIrack
from instruments.niDAQ import nidaqIO
from modules.GenericGridConstructor import gridConstructor
from CVFinder import CVFinder
import experiments.NoiseSamplingST.config_NoiseSamplingST as config

# Other imports
import numpy as np
import time
import os
from shutil import copyfile

#%% Initialization of saving config file
configSrc = config.__file__

# Initialize config object
config = config.experiment_config()
samples = config.samples
fs = config.fs
T = config.sampleTime


# Initialize data container and save directory
if config.T_test:
    saveDirectoryT = SaveLib.createSaveDirectory(config.filepath, config.name_T)
    #Tcurrents = np.zeros((samples * config.iterations, fs * T))
if config.S_test:
    saveDirectoryS = SaveLib.createSaveDirectory(config.filepath, config.name_S)
    #Scurrents = np.zeros((samples * config.iterations, fs * T))

# Initialize instruments
#ivvi = IVVIrack.initInstrument(dac_step = 500, dac_delay = 0.001)


# Find control voltages:

controlVoltages = CVFinder(config)
print(controlVoltages)
"""
# Main acquisition loop
#controlVoltages = gridConstructor(config.controls, config.steps)

if config.T_test:
    print('Testing accuracy of sample time ...')
    for i in range(0, controlVoltages.shape[0]):
        print('Getting Data for control voltage ' + str(i + 1) + ', ' + str(controlVoltages.shape[0] - i - 1) + ' control voltages remaining.')
        for j in range(samples):
            print('Sampling ' + str(j + 1) + '/' + str(samples) +'...')
            IVVIrack.setControlVoltages(ivvi, controlVoltages[i,:]) 
            time.sleep(2)  # Pause in between two samples
            Tcurrents[i * samples + j,:] = nidaqIO.IO(np.zeros(fs * T + 1), fs)

if config.S_test:
    print('Testing accuracy of switching CV config ...')
    for i in range(0, controlVoltages.shape[0]):
        print('Getting Data for control voltage ' + str(i + 1) + ', ' + str(controlVoltages.shape[0] - i - 1) + ' control voltages remaining.')
        for j in range(samples):
            print('Sampling ' + str(j + 1) + '/' + str(samples) +'...')
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
	copyfile(configSrc, config.filepath + config.name_T +'\\config_NoiseSamplingST.py')

if config.S_test:
    np.savez(os.path.join(saveDirectoryS, 'nparrays'), CV = controlVoltages, output = Scurrents)
	copyfile(configSrc, config.filepath + config.name_S + '\\config_NoiseSamplingST.py')
"""


