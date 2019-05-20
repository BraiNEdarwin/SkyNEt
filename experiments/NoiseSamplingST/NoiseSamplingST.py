# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:41:40 2018

Script to measure data for a random test set and to measure data for a noise 
analysis.

@author: Mark 
"""
# SkyNEt imports
import SkyNEt.modules.SaveLib as SaveLib
import SkyNEt.experiments.NoiseSamplingST.config_NoiseSamplingST as config
from SkyNEt.instruments import InstrumentImporter

# Other imports
import numpy as np

#%% Initialization of saving config file
configSrc = config.__file__

# Initialize config object
cf = config.experiment_config()
samples = cf.samples
fs = cf.fs
T = cf.sampleTime

# Specify whether to collect data for the random test set or to collect data for the noise fit
if cf.device == 'cDAQ':
    controlVoltages = cf.t
    print('cDAQ is used')
    if cf.experiment_type == 'test_set':
        controlVoltages = cf.CVs    
        print('Predefined control voltages are used to collect a test set')
    elif cf.experiment_type == 'noise_fit':
        controlVoltages = cf.t
        print('Noise data will be collected')
    else:
        print('Specify what experiment to do')
else:
    print('Script only supports cDAQ.')
    
# initialize save directory
saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)
data = np.zeros((samples * controlVoltages.shape[0], int(fs * T)))

# Main acquisition loop
for i in range(0, controlVoltages.shape[0]):
    print('Getting Data for control voltage ' + str(i + 1) + ', ' + str(controlVoltages.shape[0] - i - 1) + ' control voltages remaining.')
    for j in range(samples):
        print('Sampling ' + str(j + 1) + '/' + str(samples) +'...')

        if cf.device == 'cDAQ':
            # Create input data dependent on what type of experiment is chosen
            if cf.experiment_type == 'noise_fit':
                waves = cf.generateSineWave(cf.freq, cf.t[i], cf.Vmax, cf.fs_wave) \
                    * np.ones((cf.freq.shape[0], cf.sampleTime * cf.fs)) + cf.offset[:,np.newaxis]

            elif cf.experiment_type == 'test_set':
                waves = controlVoltages[i,:][:,np.newaxis] * np.ones((cf.freq.shape[0], int(cf.sampleTime * cf.fs)))

            # Add ramping up and down for the datapoints
            wavesRamped = np.zeros((waves.shape[0], waves.shape[1] + cf.rampT))
            for l in range(cf.waveElectrodes):
                try:
                    # Ramp from previous controls to current controls
                    wavesRamped[l,0:cf.rampT] = np.linspace(controlVoltages[i-1,l], waves[l,0], cf.rampT)
                except:
                    # For the first measurement a previous control doesn't exist yet: ramp from 0
                    wavesRamped[l,0:cf.rampT] = np.linspace(0,waves[l,0], cf.rampT)                  
                wavesRamped[l,cf.rampT:] = waves[l,:]

            # Actual measurement
            dataRamped = InstrumentImporter.nidaqIO.IO_cDAQ(wavesRamped, cf.fs)      
            data[i * samples + j, :] = dataRamped[:, cf.rampT:]
    
    # Save once in a while so that an error doesn't result into loss of the complete experiment
    if i%10==0:
        print('Saving...')
        SaveLib.saveExperiment(cf.configSrc, saveDirectory,
                       inputs = controlVoltages,
                       outputs = cf.amplification*data/cf.postgain,
                       fs = cf.fs,
                       electrodeSetup = cf.electrodeSetup,
                       amplification = cf.amplification,
                       gain_info = cf.gain_info,
                       rampTime = cf.rampT)

# Save obtained data 
SaveLib.saveExperiment(cf.configSrc, saveDirectory,
                       inputs = controlVoltages,
                       outputs = cf.amplification*data/cf.postgain,
                       fs = cf.fs,
                       electrodeSetup = cf.electrodeSetup,
                       amplification = cf.amplification,
                       gain_info = cf.gain_info,
                       rampTime = cf.rampT)

InstrumentImporter.reset(0,0)

