''''
Measurement script to perform an experiment generating data for NN training
'''

# Import packages
import SkyNEt.modules.SaveLib as SaveLib
from SkyNEt.instruments.niDAQ import nidaqIO
from SkyNEt.instruments import InstrumentImporter
import time
from SkyNEt.modules.GridConstructor import gridConstructor as grid
import SkyNEt.experiments.grid_search.config_grid_search as config
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

# Construct configuration array
voltages = grid(cf.electrodes, cf.voltageGrid)
voltages = voltages[:,::-1]
#voltages = np.concatenate((voltages[:,0:3], voltages[:,-2:], voltages[:,3:5]), axis = 1) # Now boolean input is on electrode 3 and 4
print('First two indices are inputs, rest CV. Fastest CV has the last index!')

# Init data container
data = np.zeros((voltages.shape[0], voltages.shape[1] + cf.samples))
data[:,:voltages.shape[1]] = voltages

# initialize save directory
saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# Initialize instruments
ivvi = InstrumentImporter.IVVIrack.initInstrument(dac_step = 500, dac_delay = 0.001)

nr_blocks = len(cf.input1)*len(cf.input2)*len(cf.input3)*len(cf.input4)
blockSize = int(len(voltages)/nr_blocks)
assert len(voltages) == blockSize*nr_blocks, 'Nr of gridpoints not divisible by nr_blocks!!'
#main acquisition loop
for j in range(nr_blocks):
    print('Getting Data for block '+str(j)+'...')
    start_block = time.time()
    InstrumentImporter.IVVIrack.setControlVoltages(ivvi, voltages[j * blockSize, :])
    time.sleep(1)  #extra delay to account for changing the input voltages
    for i in range(blockSize):
        InstrumentImporter.IVVIrack.setControlVoltages(ivvi, voltages[j * blockSize + i, :])
        time.sleep(0.175)  #tune this to avoid transients
        data[j * blockSize + i, -cf.samples:] = cf.amplification / cf.postgain * np.array(nidaqIO.IO(np.zeros(cf.samples), cf.fs))
    end_block = time.time()
    print('CV-sweep over one input state took '+str(end_block-start_block)+' sec.')

#SAVE DATA following conventions for NN training
SaveLib.saveExperiment(cf.configSrc, saveDirectory, data= data, filename = 'training_NN_data')

InstrumentImporter.reset(0,0)
