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

# config
filepath = 'D:\data\data4nn'
name = 'FullSwipe_TEST'
controlVoltages = [[-900, -600, -300, 0, 300, 600, 900]]*5
input2 = [-900, -600, -300, 0, 300, 600, 900]
input1 = [-900,0,900]
voltageGrid = [*controlVoltages,input2,input1]
electrodes = len(voltageGrid) #amount of electrodes
acqTime = 0.01 
samples = 50

#construct configuration array
voltages = grid(electrodes, voltageGrid)
voltages = voltages[:,::-1]
print('First two indices are inputs, rest CV. Fastest CV has the last index!')
# init data container
data = np.zeros((voltages.shape[0], voltages.shape[1] + samples))
data[:,:voltages.shape[1]] = voltages

# initialize save directory
saveDirectory = SaveLib.createSaveDirectory(filepath, name)
# initialize instruments
ivvi = IVVIrack.initInstrument(dac_step = 500, dac_delay = 0.001)

nr_blocks = len(input1)*len(input2)
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
np.savez(os.path.join(saveDirectory, 'training_NN_data'), data=data)

IVVIrack.setControlVoltages(ivvi, np.zeros(8))
