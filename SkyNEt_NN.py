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
# temporary imports
import numpy as np

# config
filepath = ''
name = ''
voltageGrid = [-1200, -800, -400, 0, 400, 800, 1200]
controls = 5 #amount of controls used to set voltages
acqTime = 0.01 
samples = 50
smallest_div = 7
#construct configuration array
blockSize = len(voltageGrid) ** controls
nr_blocks = 3*smallest_div

controlVoltages = Construct.CP_Grid(nr_blocks, blockSize, smallest_div, controls, voltageGrid)

# init data container
data = np.empty((controlVoltages.shape[0], samples))


# initialize save directory
saveDirectory = SaveLib.createSaveDirectory(filepath, name)

# initialize instruments
ivvi = IVVIrack.initInstrument()


#main acquisition loop
for j in range(4):
    IVVIrack.setControlVoltages(ivvi, controlVoltages[j * blockSize, :])
    time.sleep(1)  #extra delay to account for changing the input voltages
    for i in range(blocksize):
        IVVIrack.setControlVoltages(ivvi, controlVoltages[j * blockSize + i, :])
        time.sleep(0.01)  #tune this to avoid transients
        data[j * blockSize + i, :] = nidaqIO.IO(np.zeros(samples), samples/acqTime)

# some save command to finish off...

