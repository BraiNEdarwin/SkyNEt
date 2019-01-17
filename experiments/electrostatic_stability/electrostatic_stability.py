# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:51:29 2018

Scipt to test the stability of a device with electrostatic gates.
It applies various fields on the device and afterwards does a 
characterization by applying two sine waves with different frequency.

Objective is to find some voltage range which does not change the device
behaviour.

@author: Mark
@author: Bram
"""


# Import packages
import SkyNEt.modules.SaveLib as SaveLib
from SkyNEt.instruments.ADwin import adwinIO
from SkyNEt.instruments import InstrumentImporter
import time
from SkyNEt.modules.GridConstructor import gridConstructor as grid
from SkyNEt.experiments.grid_search import transient_test
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
data = np.zeros((cf.voltages.shape[0], cf.sampleTime * cf.fs))

# Construct sine waves for all grid points
waves = np.zeros((cf.waveElectrodes, cf.fs * cf.sampleTime))
t = np.arange(0, cf.sampleTime, 1 / cf.fs)
waves[0] = np.sin(2*np.pi*cf.freq[0]*t) * cf.Vmax
waves[1] = np.sin(2*np.pi*cf.freq[1]*t) * cf.Vmax

## Data acquition loop

# Set controls 3,4,5 to their voltages
InstrumentImporter.IVVIrack.setControlVoltages(ivvi, [0, 0] + cf.controlVoltages)

# For each entry in cf.voltages:
# 1. apply the voltages to C1, C2
# 2. wait cf.fieldWait seconds
# 3. apply zero volts to C1, C2
# 4. wait cf.fieldWait seconds
# 5. apply sines on in1, in2 and measure output
for i in range(voltages.shape[0]):
    # Apply voltages to C1, C2
    InstrumentImporter.IVVIrack.setControlVoltages(ivvi, voltages[i])

    # Wait for field to have some effect
    time.sleep(cf.fieldWait) 

    # Set C1, C2 back to zero
    InstrumentImporter.IVVIrack.setControlVoltages(ivvi, [0, 0])

    # Wait for field to have some effect
    time.sleep(cf.fieldWait) 

    # Apply sines and measure
    if cf.device == 'nidaq':
    	data[i] = InstrumentImporter.nidaqIO.IO(waves, cf.fs)

        # Reset to 0
        InstrumentImporter.nidaqIO.reset_device()
    elif cf.device == 'adwin':
    	data[i*cf.sampleTime*cf.fs: (i+1)*cf.sampleTime*cf.fs] = InstrumentImporter.adwinIO.IO(adwin, waves[:, i*cf.sampleTime*cf.fs: (i+1)*cf.sampleTime*cf.fs], cf.fs)
    else:
    	print("Please specify device")

SaveLib.saveExperiment(saveDirectory, 
                       voltages = cf.voltages*5,  # mV, compensate for gain
                       waves = waves, 
                       t = t,
                       output = data*cf.amplification, 
                       filename = 'data')
    

InstrumentImporter.reset(0,0)
adwinIO.reset(adwin)

