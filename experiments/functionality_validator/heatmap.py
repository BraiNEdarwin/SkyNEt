# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:51:29 2018
Script to measure (2D) input voltages to make a heatmap of the input space
functions
@author: Mark
"""
# Import packages
import SkyNEt.modules.SaveLib as SaveLib
from SkyNEt.instruments import InstrumentImporter
import SkyNEt.experiments.functionality_validator.config_heatmap as config
# temporary imports
import numpy as np
import time

#%% Initialization of saving config file
configSrc = config.__file__

# Initialize config object
cf = config.experiment_config()

# initialize save directory
saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# Initialize output data set
data = np.zeros((cf.controlVoltages.shape[0], int(cf.sampleTime * cf.fs)))

batches = int(cf.fs * cf.sampleTime / cf.samplePoints)

for gates in range(cf.controlVoltages.shape[0]):
    print('Collecting for gate ' + str(gates + 1))
    for i in range(0, batches):
        start_wave = time.time()
        
        t = np.linspace(i * cf.samplePoints, (i + 1) * cf.samplePoints - 1, cf.samplePoints)
        waves = cf.generateSineWave(cf.freq, t, cf.amplitude, cf.fs, cf.phase) + np.outer(cf.offset, np.ones(t.shape[0]))
        controls = cf.controlVoltages[gates,:][:,np.newaxis] * np.ones((cf.controlElectrodes, cf.samplePoints))
        
        # Append input waves and static controls
        inputs = np.insert(controls, [cf.inputIndex[0], cf.inputIndex[1]-1], waves, axis=0)

        # Use 0.5 second to ramp up to the value where data aqcuisition stopped previous iteration
        # and 0.5 second to ramp down after the batch is done

        inputsRamped = np.zeros((inputs.shape[0], inputs.shape[1] + int(cf.fs))) 
        dataRamped = np.zeros(inputsRamped.shape[1])

        for j in range(inputsRamped.shape[0]):
            inputsRamped[j,0:int(0.5*cf.fs)] = np.linspace(0,inputs[j,0], int(0.5*cf.fs))
            inputsRamped[j,int(0.5*cf.fs): int(0.5*cf.fs) + inputs.shape[1]] = inputs[j,:]
            inputsRamped[j,int(0.5*cf.fs) + inputs.shape[1]:] = np.linspace(inputs[j,-1], 0, int(0.5*cf.fs))
            
        dataRamped = InstrumentImporter.nidaqIO.IO_cDAQ(inputsRamped, cf.fs)      
        data[gates, i*cf.samplePoints: (i+1)*cf.samplePoints] = dataRamped[0,int(0.5*cf.fs):int(0.5*cf.fs) + inputs.shape[1]]

        if i % 10 == 0: # Save after every 10 mini batches
            print('Saving...')
            SaveLib.saveExperiment(cf.configSrc, saveDirectory, 
                                    output = data*cf.amplification/cf.postgain,
                                    freq = cf.freq,
                                    sampleTime = cf.sampleTime,
                                    fs = cf.fs,
                                    phase = cf.phase,
                                    amplitude = cf.amplitude,
                                    offset = cf.offset,
                                    filename = 'nparrays')
        end_wave = time.time()
        print('Data collection for part ' + str(i+1) + ' of ' + str(batches) + ' took '+str(end_wave-start_wave)+' sec.')

    
    

SaveLib.saveExperiment(cf.configSrc, saveDirectory, 
                        output = data*cf.amplification/cf.postgain,
                        freq = cf.freq,
                        sampleTime = cf.sampleTime,
                        fs = cf.fs,
                        phase = cf.phase,
                        amplitude = cf.amplitude,
                        offset = cf.offset,
                        filename = 'nparrays')
  

InstrumentImporter.reset(0,0)