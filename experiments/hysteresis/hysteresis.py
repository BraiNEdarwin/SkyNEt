# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:51:29 2018

Script to check whether we have hysteresis in the devices

@author: Mark
"""
# Import packages
import SkyNEt.modules.SaveLib as SaveLib
from SkyNEt.instruments import InstrumentImporter
import time
import SkyNEt.experiments.hysteresis.config_hysteresis as config
# temporary imports
import numpy as np

#%% Initialization of saving config file
configSrc = config.__file__

# Initialize config object
cf = config.experiment_config()

# initialize save directory
saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

t = np.arange(0,cf.samplePoints)
waves = cf.generateSineWave(cf.freq, t, cf.amplitude, cf.fs, cf.phase) + np.outer(cf.offset, np.ones(t.shape[0]))
data = np.zeros((waves.shape[1], cf.nr_halfloops))

# Use 0.5 second to ramp up to the value where data aqcuisition stopped previous iteration
# and 0.5 second to ramp down after the batch is done
wavesRamped = np.zeros((waves.shape[0], waves.shape[1] + int(cf.fs))) 
dataRamped = np.zeros(wavesRamped.shape[1])
for j in range(wavesRamped.shape[0]):
    wavesRamped[j,0:int(0.5*cf.fs)] = np.linspace(0,waves[j,0], int(0.5*cf.fs))
    wavesRamped[j,int(0.5*cf.fs): int(0.5*cf.fs) + waves.shape[1]] = waves[j,:]
    wavesRamped[j,int(0.5*cf.fs) + waves.shape[1]:] = np.linspace(waves[j,-1], 0, int(0.5*cf.fs))

print('Starting measurement, ' + str(int(cf.samplePoints/cf.fs)) + ' seconds per minibatch')
for i in range(cf.nr_halfloops):
    # Every second batch must be flipped to go back and forth
    if True:
        wavesRamped = wavesRamped[:,::-1]

    start_wave = time.time()
      
    dataRamped = InstrumentImporter.nidaqIO.IO_cDAQ(wavesRamped, cf.fs)      
    data[:, i] = dataRamped[0,int(0.5*cf.fs):int(0.5*cf.fs) + waves.shape[1]]

    end_wave = time.time()
    print('Data collection for part ' + str(i+1) + ' of ' + str(cf.nr_halfloops) + ' took '+str(end_wave-start_wave)+' sec.')

    
    
SaveLib.saveExperiment(cf.configSrc, saveDirectory, 
                            intputs = waves,
                            output = data*cf.amplification/cf.postgain,
                            freq = cf.freq,
                            sampleTime = cf.sampleTime,
                            fs = cf.fs,
                            phase = cf.phase,
                            amplitude = cf.amplitude,
                            offset = cf.offset,
                            amplification = cf.amplification,
                            electrodeSetup = cf.electrodeSetup,
                            gain_info = cf.gain_info,
                            nr_loops = cf.nr_halfloops/2,
                            filename = 'data')
      

InstrumentImporter.reset(0,0)