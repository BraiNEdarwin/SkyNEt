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
from SkyNEt.intruments import InstrumentImporter
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
adwin = adwinIO.initInstrument()

# Construct configuration array
voltages = grid(cf.gridElectrodes, cf.controlVoltages)
data = np.zeros((voltages.shape[0], cf.sampleTime * cf.fs))

# Construct sine waves for all grid points
waves = np.zeros((voltages.shape[0] * cf.waveElectrodes, cf.fs * cf.sampleTime))
for i in range(voltages.shape[0]):
    t = np.arange(cf.sampleTime*i, cf.sampleTime * (i + 1), 1 / cf.fs)
    waves[i * cf.waveElectrodes:(i + 1) * cf.waveElectrodes, :] = np.sin(cf.freq[:,np.newaxis] * t[np.newaxis] + cf.phase) *cf.Vmax


#main acquisition loop
#TODO: ramp slowly up to the first data point to avoid high transient at the start
for i in range(voltages.shape[0]):
    start_wave = time.time()
    InstrumentImporter.IVVIrack.setControlVoltages(ivvi, voltages[i, :])
    time.sleep(0.3)
    data[i,:] = adwinIO.IO(adwin, waves[i*cf.waveElectrodes: (i+1)*cf.waveElectrodes], cf.fs)
    end_wave = time.time()
    print('CV-sweep over grid point ' + str(i) + ' of ' +  str(voltages.shape[0]) + ' took '+str(end_wave-start_wave)+' sec.')
    
    
SaveLib.saveExperiment(cf.configSrc, saveDirectory, data= data, filename = 'training_NN_data')

InstrumentImporter.reset(0,0)
adwinIO.reset()
