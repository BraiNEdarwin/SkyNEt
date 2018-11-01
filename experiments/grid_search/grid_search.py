''''
Measurement script to perform an experiment generating data for NN training
'''

# Import packages
import SkyNEt.modules.SaveLib as SaveLib
from SkyNEt.instruments.niDAQ import nidaqIO
from SkyNEt.instruments.DAC import IVVIrack
import time
from SkyNEt.modules.GridConstructor import gridConstructor as grid
# temporary imports
import numpy as np
import os

# Initialize config object
cf = config.experiment_config()

# Construct configuration array
voltages = grid(cf.electrodes, cf.voltageGrid)
voltages = voltages[:,::-1]
print('First two indices are inputs, rest CV. Fastest CV has the last index!')

# Init data container
data = np.zeros((voltages.shape[0], voltages.shape[1] + cf.samples))
data[:,:voltages.shape[1]] = voltages

# initialize save directory
saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# Initialize instruments
ivvi = IVVIrack.initInstrument(dac_step = 500, dac_delay = 0.001)

nr_blocks = len(cf.input1)*len(cf.input2)
blockSize = int(len(voltages)/nr_blocks)
assert len(voltages) == blockSize*nr_blocks, 'Nr of gridpoints not divisible by nr_blocks!!'
#main acquisition loop
for j in range(nr_blocks):
    print('Getting Data for block '+str(j)+'...')
    start_block = time.time()
    IVVIrack.setControlVoltages(ivvi, voltages[j * blockSize, :])
    time.sleep(1)  #extra delay to account for changing the input voltages
    for i in range(blockSize):
        IVVIrack.setControlVoltages(ivvi, voltages[j * blockSize + i, :])
        time.sleep(0.01)  #tune this to avoid transients
        data[j * blockSize + i, -samples:] = nidaqIO.IO(np.zeros(samples+1), samples/acqTime)
    end_block = time.time()
    print('CV-sweep over one input state took '+str(end_block-start_block)+' sec.')

#SAVE DATA following conventions for NN training
SaveLib.saveExperiment(saveDirectory, data=data, filename = 'training_NN_data')
# np.savez(os.path.join(saveDirectory, 'training_NN_data'), data=data)

ivvi.set_dacs_zero()
