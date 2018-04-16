# This file performs a transient experiment. I.e. change some controls and record the transient response


# Import packages
import modules.ReservoirFull as Reservoir
import modules.PlotBuilder as PlotBuilder
import modules.GenerateInput as GenerateInput
import modules.Evolution as Evolution
import modules.PostProcess as PostProcess
import modules.SaveLib as SaveLib
from instruments.niDAQ import nidaqIO
from instruments.DAC import IVVIrack

# temporary imports
import numpy as np

t = 2
Fs = 1000

ivvi = IVVIrack.initInstrument(dac_step = 2000, dac_delay = 0.01)

#output initialization
outputs = np.empty((21*7, t*Fs))

#define controlvoltages
controlVoltages = np.zeros((21 * 7, 7))

for i in range(21):
    controlVoltages[i,:] = [-2000 + i*200, 0, 0, 0, 0, 0, 0]

controlVoltages[21:42, 0] = controlVoltages[0:21, 0]
controlVoltages[21:42, 1] = controlVoltages[0:21, 0]

controlVoltages[42:63, 0] = controlVoltages[0:21, 0]
controlVoltages[42:63, 1] = controlVoltages[0:21, 0]
controlVoltages[42:63, 2] = controlVoltages[0:21, 0]

controlVoltages[63:84, 0] = controlVoltages[0:21, 0]
controlVoltages[63:84, 1] = controlVoltages[0:21, 0]
controlVoltages[63:84, 2] = controlVoltages[0:21, 0]
controlVoltages[63:84, 3] = controlVoltages[0:21, 0]

controlVoltages[84:105, 0] = controlVoltages[0:21, 0]
controlVoltages[84:105, 1] = controlVoltages[0:21, 0]
controlVoltages[84:105, 2] = controlVoltages[0:21, 0]
controlVoltages[84:105, 3] = controlVoltages[0:21, 0]
controlVoltages[84:105, 4] = controlVoltages[0:21, 0]

controlVoltages[105:126, 0] = controlVoltages[0:21, 0]
controlVoltages[105:126, 1] = controlVoltages[0:21, 0]
controlVoltages[105:126, 2] = controlVoltages[0:21, 0]
controlVoltages[105:126, 3] = controlVoltages[0:21, 0]
controlVoltages[105:126, 4] = controlVoltages[0:21, 0]
controlVoltages[105:126, 5] = controlVoltages[0:21, 0]

controlVoltages[126:147, 0] = controlVoltages[0:21, 0]
controlVoltages[126:147, 1] = controlVoltages[0:21, 0]
controlVoltages[126:147, 2] = controlVoltages[0:21, 0]
controlVoltages[126:147, 3] = controlVoltages[0:21, 0]
controlVoltages[126:147, 4] = controlVoltages[0:21, 0]
controlVoltages[126:147, 5] = controlVoltages[0:21, 0]
controlVoltages[126:147, 6] = controlVoltages[0:21, 0]

#start reading signal for set time

y = np.zeros(t*Fs)


for i in range(21*7):
    outputs[i, :] = nidaqIO.IO(y, Fs)
    
    #set control voltages
    IVVIrack.setControlVoltages(ivvi, controlVoltages[i, :])






