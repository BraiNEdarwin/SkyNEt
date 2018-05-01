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
import modules.NeuralNetTraining as NN
from instruments.niDAQ import nidaqIO
from instruments.DAC import IVVIrack
import time

# temporary imports
import numpy as np

# config
filepath = ''
name = ''
voltageGrid = [-2000, -1600, -1200, -800, -400, 0, 400, 800, 1200, 1600, 2000]
controls = 5 #amount of controls used to set voltages
acqTime = 0.01 
samples = 50

#construct configuration array
blockSize = len(voltageGrid) ** controls
controlVoltages = np.empty((4 * blockSize, 7))

controlVoltages[0:blockSize,0] = 0
controlVoltages[blockSize:2*blockSize,0] = 1
controlVoltages[2*blockSize:3*blockSize,0] = 0
controlVoltages[3*blockSize:4*blockSize,0] = 1

controlVoltages[0:blockSize,1] = 0
controlVoltages[blockSize:2*blockSize,1] = 0
controlVoltages[2*blockSize:3*blockSize,1] = 1
controlVoltages[3*blockSize:4*blockSize,1] = 1

controlVoltages[0:blockSize, 2:2+controls] = NN.initTraj(controls, voltageGrid)
controlVoltages[blockSize:2*blockSize, 2:2+controls] = NN.initTraj(controls, voltageGrid)
controlVoltages[2*blockSize:3*blockSize, 2:2+controls] = NN.initTraj(controls, voltageGrid)
controlVoltages[3*blockSize:4*blockSize, 2:2+controls] = NN.initTraj(controls, voltageGrid)

# init data container
data = np.empty((4 * blockSize, samples))


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

