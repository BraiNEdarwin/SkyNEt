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
from SkyNEt.experiments.wave_search import transient_test
import SkyNEt.experiments.wave_search.config_wave_search as config
# temporary imports
import numpy as np

#%% Initialization of saving config file
configSrc = config.__file__

# Initialize config object
cf = config.experiment_config()

# initialize save directory
saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# Initialize output data set
data = np.zeros((1, cf.sampleTime * cf.fs))

# Option to load the input data in small parts
if cf.loadData:
    n_loads = int(cf.fs * cf.sampleTime / cf.loadPoints) # Number of times to load parts of the input
      
    for i in range(0, n_loads):
        start_wave = time.time()
        
        waves = np.load(cf.loadString)['waves'][:,i*cf.loadPoints: (i+1)*cf.loadPoints]
        # Use 1 second to ramp up to the value where data aqcuisition stopped previous iteration
        wavesRamped = np.zeros((waves.shape[0], waves.shape[1] + cf.fs)) 
        dataRamped = np.zeros((1,wavesRamped.shape[1]))
        for j in range(wavesRamped.shape[0]):
            wavesRamped[j,0:cf.fs] = np.linspace(0,waves[j,0], cf.fs)
            wavesRamped[j,cf.fs:] = waves[j,:]
            
        dataRamped = InstrumentImporter.nidaqIO.IO_cDAQ(wavesRamped, cf.fs)      
        data[0, i*cf.loadPoints: (i+1)*cf.loadPoints] = dataRamped[:, cf.fs:]
        end_wave = time.time()
        print('Data collection for part ' + str(i+1) + ' of ' + str(n_loads) + ' took '+str(end_wave-start_wave)+' sec.')
        
    # The last batch size is variable to the sample time and the amount of loadPoints used
    if (cf.fs * cf.sampleTime - n_loads * cf.loadPoints) > 0:
	    print('Last batch size: ' + str(cf.fs * cf.sampleTime - n_loads * cf.loadPoints))
	    waves = np.load(cf.loadString)['waves'][:,n_loads*cf.loadPoints:]
	    wavesRamped = np.zeros((waves.shape[0], waves.shape[1] + cf.fs)) 
	    dataRamped = np.zeros((1,wavesRamped.shape[1]))
	    for j in range(wavesRamped.shape[0]):
	        wavesRamped[j,0:cf.fs] = np.linspace(0,waves[j,0], cf.fs)
	        wavesRamped[j,cf.fs:] = waves[j,:]
	    dataRamped = InstrumentImporter.nidaqIO.IO_cDAQ(wavesRamped, cf.fs)      
	    data[0, n_loads*cf.loadPoints:] = dataRamped[:, cf.fs:]  
    
else:
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
    if cf.loadData:
        print("Only for the last loaded data transients are tested for convenience")
        ytestdata, difference, xtestdata = transient_test.transient_test(waves, data[0, n_loads*cf.loadPoints:], cf.fs, cf.sampleTime, cf.n)
        SaveLib.saveExperiment(cf.configSrc, saveDirectory, xtestdata = xtestdata, ytestdata = ytestdata*cf.amplification/cf.postgain, \
                               diff = difference*cf.amplification/cf.postgain, \
                               output = data*cf.amplification/cf.postgain, filename = 'training_NN_data')
    else:   
        ytestdata, difference, xtestdata = transient_test.transient_test(waves, data, cf.fs, cf.sampleTime, cf.n)
    SaveLib.saveExperiment(cf.configSrc, saveDirectory, xtestdata = xtestdata, ytestdata = ytestdata*cf.amplification/cf.postgain, \
                           diff = difference*cf.amplification/cf.postgain, \
                           waves = waves, output = data*cf.amplification/cf.postgain, filename = 'training_NN_data')
elif cf.loadData:
    SaveLib.saveExperiment(cf.configSrc, saveDirectory, output = data*cf.amplification/cf.postgain, filename = 'training_NN_data')
else:
    SaveLib.saveExperiment(cf.configSrc, saveDirectory, waves = waves, output = data*cf.amplification/cf.postgain, filename = 'training_NN_data')
    

InstrumentImporter.reset(0,0)

