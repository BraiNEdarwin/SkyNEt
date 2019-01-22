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
from SkyNEt.experiments.wave_search import transient_test
import SkyNEt.experiments.wave_search.config_wave_search as config
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
adwin = InstrumentImporter.adwinIO.initInstrument()

# Initialize output data set
data = np.zeros((1, cf.sampleTime * cf.fs))

# Construct sine waves for all grid points
waves = np.zeros((cf.waveElectrodes, cf.fs * cf.sampleTime))
t = np.arange(0, cf.sampleTime, 1 / cf.fs)
waves = np.sin(2* np.pi * cf.freq[:,np.newaxis] * t[np.newaxis]) *cf.Vmax  #\\ Note that it starts at phase 0


#main acquisition
start_wave = time.time()
data = InstrumentImporter.nidaqIO.IO_cDAQ(waves, cf.fs)
end_wave = time.time()
print('Data collection took '+str(end_wave-start_wave)+' sec.')

if cf.transientTest:
    print("Testing for transients...")
    ytestdata, difference, xtestdata = transient_test.transient_test(waves, data, cf.fs, cf.sampleTime, cf.n)
    SaveLib.saveExperiment(cf.configSrc, saveDirectory, xtestdata = xtestdata, ytestdata = ytestdata*cf.amplification/cf.postgain, diff = difference*cf.amplification/cf.postgain, \
        waves = waves, output = data*cf.amplification/cf.postgain, filename = 'training_NN_data')
else:
    SaveLib.saveExperiment(cf.configSrc, saveDirectory, waves = waves, output = data*cf.amplification/cf.postgain, filename = 'training_NN_data')
    

InstrumentImporter.reset(0,0)
adwinIO.reset(adwin)

