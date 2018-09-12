''''
Measurement script to perform an experiment generating data for NN training
'''

# Import packages
import modules.ReservoirFull as Reservoir
import modules.PlotBuilder as PlotBuilder
import modules.GenerateInput as GenerateInput
import modules.Evolution as Evolution
import modules.PostProcess as PostProcess
import modules.SaveLib as SaveLib
from instruments.niDAQ import nidaqIO
from instruments.DAC import IVVIrack
import time
import modules.Grid_Constructor as Construct
import modules.NeuralNetTraining as NN
# temporary imports
import numpy as np
import os
# config


filepath = 'D:\data\Hans\\'
name = 'CP_FullSwipe'
voltageGrid = [-350, -225, -75, 75, 225, 350, 450]

controls = 5 #amount of controls used to set voltages
acqTime = 0.01 
samples = 50
smallest_div = 7

biggest_div = 3
#construct configuration array
blockSize = len(voltageGrid) ** controls

nr_blocks = biggest_div*smallest_div

controlVoltages = Construct.CP_Grid(nr_blocks, blockSize, smallest_div, controls, voltageGrid)

# init data container


data = np.zeros((controlVoltages.shape[0], controlVoltages.shape[1] + samples))
data[:,:controlVoltages.shape[1]] = controlVoltages



# initialize save directory
saveDirectory = SaveLib.createSaveDirectory(filepath, name)


# initialize instruments
ivvi = IVVIrack.initInstrument(dac_step = 500, dac_delay = 0.001)


#main acquisition loop
for j in range(nr_blocks):
    print('Getting Data for block '+str(j)+'...')
    start_fullSweep = time.time()
    IVVIrack.setControlVoltages(ivvi, controlVoltages[j * blockSize, :])
    time.sleep(1)  #extra delay to account for changing the input voltages
    for i in range(blockSize):

        IVVIrack.setControlVoltages(ivvi, controlVoltages[j * blockSize + i, :])
        time.sleep(0.01)  #tune this to avoid transients
        data[j * blockSize + i, -samples:] = nidaqIO.IO(np.zeros(samples+1), samples/acqTime)
        if j == 0 and i<25: print('Data acquired: ',data[j * blockSize + i, -samples:])
    end_fullSweep = time.time()
    print('CV-sweep over one input state took '+str(end_fullSweep-start_fullSweep)+' sec.')
# some save command to finish off...
#data = np.concatenate((controlVoltages,data),axis=1)
np.savez(os.path.join(saveDirectory, 'nparrays'), data=data)

IVVIrack.setControlVoltages(ivvi, np.zeros(8))

