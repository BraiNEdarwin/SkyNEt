# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:51:29 2018
Script to characterize a device using partially a grid and partially sine wave 
functions
@author: Mark
"""
# Import packages
import SkyNEt.modules.SaveLib as SaveLib
from SkyNEt.instruments import InstrumentImporter
import time
import SkyNEt.experiments.crosstalk_test.config_crosstalk_test as config
# temporary imports
import numpy as np

#%% Initialization of saving config file
configSrc = config.__file__

# Initialize config object
cf = config.experiment_config()

# initialize save directory
saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)


data = np.zeros((cf.factor_gain.shape[0],int(cf.sampleTime * cf.fs)))

t = np.linspace(0, cf.fs*cf.sampleTime - 1, cf.fs*cf.sampleTime) # Reduce T with increasing f, total samples should stay the same
waves = cf.inputData(cf.freq, t, cf.amplitude, cf.fs, cf.phase) + np.outer(cf.offset, np.ones(t.shape[0]))


for i in range(cf.factor_gain.shape[0]):
    # Initialize output data set   

    # Use 2 seconds to ramp up to the value where data aqcuisition stopped previous iteration
    # and 2 seconds to ramp down after the batch is done
    wavesRamped = np.zeros((waves.shape[0], waves.shape[1] + 4*int(cf.fs * cf.factor_gain[i]))) 
    dataRamped = np.zeros(wavesRamped.shape[1])
    for j in range(wavesRamped.shape[0]):
        wavesRamped[j,0:int(2*cf.fs *cf.factor_gain[i])] = np.linspace(0,waves[j,0], int(2*cf.fs * cf.factor_gain[i]))
        wavesRamped[j,int(2*cf.fs * cf.factor_gain[i]): int(2*cf.fs * cf.factor_gain[i]) + waves.shape[1]] = waves[j,:]
        wavesRamped[j,int(2*cf.fs * cf.factor_gain[i]) + waves.shape[1]:] = np.linspace(waves[j,-1], 0, int(2*cf.fs * cf.factor_gain[i]))

    print('Measuring sine waves for factor ' + str(cf.factor_gain[i]))   
    
        
    dataRamped = InstrumentImporter.nidaqIO.IO_cDAQ(wavesRamped, cf.fs * cf.factor_gain[i])      
    data[i,:] = dataRamped[0,int(2*cf.fs * cf.factor_gain[i]):int(2*cf.fs * cf.factor_gain[i]) + waves.shape[1]]

        
        

    print('Saving...')
    SaveLib.saveExperiment(cf.configSrc, saveDirectory, 
                            output = data*cf.amplification/cf.postgain,
                            freq = cf.freq,
                            factor_gain = cf.factor_gain,
                            sampleTime = cf.sampleTime,
                            fs = cf.fs,
                            phase = cf.phase,
                            amplitude = cf.amplitude,
                            offset = cf.offset,
                            amplification = cf.amplification,
                            electrodeSetup = cf.electrodeSetup,
                            gain_info = cf.gain_info,
                            filename = 'data')
      

InstrumentImporter.reset(0,0)