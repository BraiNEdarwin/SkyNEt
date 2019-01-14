# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:51:29 2018

Script to characterize a device using partially a grid and partially sine wave 
functions
@author: Mark
"""


# Import packages
import SkyNEt.modules.SaveLib as SaveLib
from SkyNEt.instruments.ADwin import adwinIO
from SkyNEt.instruments import InstrumentImporter
import time
from SkyNEt.modules.GridConstructor import gridConstructor as grid
import SkyNEt.experiments.grid_search.config_wave_grid as config
# temporary imports
import numpy as np
import os
import signal
import sys
from shutil import copyfile

#%% Initialization of saving config file
configSrc = config.__file__

# Initialize config object
cf = config.experiment_config()

# initialize save directory
saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# Initialize instruments
ivvi = InstrumentImporter.IVVIrack.initInstrument(dac_step = 500, dac_delay = 0.001)
adwin = InstrumentImporter.adwinIO.initInstrument()

# Construct configuration array
voltages = grid(cf.gridElectrodes, cf.controlVoltages)
data = np.zeros((1, voltages.shape[0] * cf.sampleTime * cf.fs))

# Construct sine waves for all grid points
waves = np.zeros((cf.waveElectrodes, voltages.shape[0] * cf.fs * cf.sampleTime))
t = np.arange(0, voltages.shape[0] * cf.sampleTime, 1 / cf.fs)
waves = np.sin(cf.freq[:,np.newaxis] * t[np.newaxis]) *cf.Vmax  #\\ Note that it starts at phase 0


#main acquisition loop
#TODO: ramp slowly up to the first data point to avoid high transient at the start
for i in range(voltages.shape[0]):
    start_wave = time.time()
    InstrumentImporter.IVVIrack.setControlVoltages(ivvi, voltages[i, :])
    time.sleep(0.3) # Pause to avoid transients
    if cf.device == 'nidaq':
    	data[i*cf.sampleTime*cf.fs: (i+1)*cf.sampleTime*cf.fs] = InstrumentImporter.nidaqIO.IO(waves[:, i*cf.sampleTime*cf.fs: (i+1)*cf.sampleTime*cf.fs], cf.fs)
    elif cf.device == 'adwin':
    	data[i*cf.sampleTime*cf.fs: (i+1)*cf.sampleTime*cf.fs] = InstrumentImporter.adwinIO.IO(adwin, waves[:, i*cf.sampleTime*cf.fs: (i+1)*cf.sampleTime*cf.fs], cf.fs)
    else:
    	print("Please specify device")
    end_wave = time.time()
    print('CV-sweep over grid point ' + str(i) + ' of ' +  str(voltages.shape[0]) + ' took '+str(end_wave-start_wave)+' sec.')

########################################
# Test part for testing for transients #
########################################
print('Transient test')
n = 10 	# Amount of datapoints to be tested
testdata = np.zeros((n, 2*cf.fs))
test_cases = np.random.randint(voltages.shape[0], size=n)
difference = np.zeros((n,1))
for i in range(n):     
    InstrumentImporter.IVVIrack.setControlVoltages(ivvi, voltages[test_cases[i], :])    
    time.sleep(1)
    if cf.device == 'nidaq':
        testdata[i,:] = InstrumentImporter.nidaqIO.IO(np.ones((waves.shape[0], 2*cf.fs)) * waves[:, cf.fs * test_cases[i],np.newaxis], cf.fs) # sample for 2s
    elif cf.device == 'adwin':
    	testdata[i,:] = InstrumentImporter.adwinIO.IO(adwin, np.ones((waves.shape[0], 2*cf.fs)) * waves[:, cf.fs * test_cases[i],np.newaxis], cf.fs)
    difference[i,0] = np.mean(testdata[i,:]) - data[0,cf.fs * test_cases[i]]
  		

SaveLib.saveExperiment(cf.configSrc, saveDirectory, testdata = testdata*cf.amplification/cf.postgain, diff = difference*cf.amplification/cf.postgain, \
	grid = voltages, waves = waves, output = data*cf.amplification/cf.postgain, filename = 'training_NN_data')

InstrumentImporter.reset(0,0)
adwinIO.reset(adwin)
