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
import os

# temporary imports
import numpy as np

# config
filepath = 'D:\\data\\BramdW\\nndata'
name = 'dataset_sample180329_43'
voltageGrid = [-1000, -800, -600, -400, -200, 0, 200, 400, 600, 800, 1000]
controls = 5 #amount of controls used to set voltages
acqTime = 0.1
samples = 100

#construct configuration array
blockSize = len(voltageGrid) ** controls
controlVoltages = np.empty((4 * blockSize, 7))

controlVoltages[0:blockSize,0] = 0
controlVoltages[blockSize:2*blockSize,0] = 0.5
controlVoltages[2*blockSize:3*blockSize,0] = 0
controlVoltages[3*blockSize:4*blockSize,0] = 0.5

controlVoltages[0:blockSize,1] = 0
controlVoltages[blockSize:2*blockSize,1] = 0
controlVoltages[2*blockSize:3*blockSize,1] = 0.5
controlVoltages[3*blockSize:4*blockSize,1] = 0.5

controlVoltages[0:blockSize, 2:2+controls] = NN.initTraj(controls, voltageGrid)
controlVoltages[blockSize:2*blockSize, 2:2+controls] = NN.initTraj(controls, voltageGrid)
controlVoltages[2*blockSize:3*blockSize, 2:2+controls] = NN.initTraj(controls, voltageGrid)
controlVoltages[3*blockSize:4*blockSize, 2:2+controls] = NN.initTraj(controls, voltageGrid)

# init data container
data = np.empty((4 * blockSize, 9))


# initialize save directory
saveDirectory = SaveLib.createSaveDirectory(filepath, name)

# initialize instruments
ivvi = IVVIrack.initInstrument(dac_step = 500, dac_delay = 0.001)


#main acquisition loop
for j in range(4):
    IVVIrack.setControlVoltages(ivvi, controlVoltages[j * blockSize, :])
    time.sleep(1)  #extra delay to account for changing the input voltages
    for i in range(blockSize):
        if(i != 0):
        	for k in range(controls):
        		if(controlVoltages[j * blockSize + i, k] != controlVoltages[j * blockSize + i - 1, k]):	
        		    IVVIrack.setControlVoltage(ivvi, controlVoltages[j * blockSize + i, k], k)
        		    break
        output = nidaqIO.IO(np.zeros(samples), samples/acqTime)
        data[j * blockSize + i, 7] = np.mean(output)
        data[j * blockSize + i, 8] = np.std(output)
    print('Block ' + str(j + 1) + '/4 completed')
 
data[0:4*blockSize, 0:7] = controlVoltages    
# some save command  to finish off...
np.savez(os.path.join(saveDirectory, 'nparrays'), data = data)

IVVIrack.setControlVoltages(ivvi, np.zeros(8))
