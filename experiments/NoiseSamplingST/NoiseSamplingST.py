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
from SkyNEt.modules.GridConstructor import gridConstructor
from CVFinder import CVFinder
import experiments.NoiseSamplingST.config_NoiseSamplingST as config
from SkyNEt.instruments import InstrumentImporter
import matplotlib.pyplot as plt

# Other imports
import numpy as np
import time
import os
from shutil import copyfile

#%% Initialization of saving config file
configSrc = config.__file__

# Initialize config object
cf = config.experiment_config()
samples = cf.samples
fs = cf.fs
T = cf.sampleTime

# Initialize instrument
ivvi = IVVIrack.initInstrument(dac_step = 500, dac_delay = 0.001)



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

# Initialize data container and save directory

saveDirectoryT = SaveLib.createSaveDirectory(cf.filepath, cf.name_T)
Tcurrents = np.zeros((samples * controlVoltages.shape[0], int(fs * T)))

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
            wavesRamped = np.zeros((waves.shape[0], waves.shape[1] + 2*cf.rampT))
            for l in range(cf.waveElectrodes):

                wavesRamped[l,0:cf.rampT] = np.linspace(0,waves[l,0], cf.rampT)
                wavesRamped[l,cf.rampT:cf.rampT+waves.shape[1]] = waves[l,:]
                wavesRamped[l,cf.rampT+waves.shape[1]:] = np.linspace(waves[l,-1],0, cf.rampT)

            # Actual measurement
            dataRamped = InstrumentImporter.nidaqIO.IO_cDAQ(wavesRamped, cf.fs)      
            Tcurrents[i * samples + j, :] = dataRamped[:, cf.rampT:cf.rampT+waves.shape[1]]
    if i%10==0:
        print('Saving...')
        np.savez(os.path.join(saveDirectoryT, 'nparrays'), CV = controlVoltages, output = cf.amplification*Tcurrents/cf.postgain)

        #else:
        #    wavesRamped = np.zeros((cf.waveElectrodes, int(fs * T + 12*cf.rampT)))
        #    for l in range(cf.waveElectrodes):

        #        wavesRamped[l,0:cf.rampT] = np.linspace(0,cf.CVs[i,l], cf.rampT)
        #        wavesRamped[l,cf.rampT:cf.rampT+int(fs * T)] = cf.CVs[i,l] * np.ones(cf.sampleTime * cf.fs)
        #            wavesRamped[l,cf.rampT+int(fs * T):2*cf.rampT+int(fs * T)] = np.linspace(cf.CVs[i,l],0, cf.rampT)

        #        dataRamped = InstrumentImporter.nidaqIO.IO_cDAQ(wavesRamped, cf.fs)      
        #        Tcurrents[i * samples + j, :] = dataRamped
                #IVVIrack.setControlVoltages(ivvi, controlVoltages[i,:]) 
                #time.sleep(0.5)  # Pause in between two samples
                #Tcurrents[i * samples + j,:] = nidaqIO.IO(np.zeros(fs * T), fs)
            
# Set IVVI back to zero
IVVIrack.setControlVoltages(ivvi, np.zeros(8))   


# Save obtained data (the two tests are saved in separate files)

np.savez(os.path.join(saveDirectoryT, 'nparrays'), CV = controlVoltages, output = cf.amplification*Tcurrents/cf.postgain)
copyfile(configSrc, saveDirectoryT +'\\config_NoiseSamplingST.py') # TODO: fix bug with configSrc

InstrumentImporter.reset(0,0)

