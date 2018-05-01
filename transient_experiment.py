# This file performs a transient experiment. I.e. change some controls and record the transient response
# can at present not be done with DACs, because they block the script...


# Import packages
import modules.ReservoirFull as Reservoir
import modules.PlotBuilder as PlotBuilder
import modules.GenerateInput as GenerateInput
import modules.Evolution as Evolution
import modules.PostProcess as PostProcess
import modules.SaveLib as SaveLib
from instruments.niDAQ import nidaqIO
from instruments.DAC import IVVIrack
import os
import time
from multiprocessing import Process

from qcodes.instrument_drivers.QuTech.IVVI import IVVI
import nidaqmx

# temporary imports
import numpy as np


t = 2
Fs = 1000

#ivvi = IVVIrack.initInstrument(dac_step = 2000, dac_delay = 0.01)
inputs = np.empty((21, t*Fs))
inputmodel = np.empty(t*Fs)
inputmodel[0:round(t*Fs/2)] = 0
inputmodel[round(t*Fs/2):] = 1
inputmodel[-1] = 0

for i in range(21):
	inputs[i, :] = inputmodel * (-2 + i*0.2)

#output initialization
outputs = np.empty((21, t*Fs))

for i in range(21):
	outputs[i,:] = nidaqIO.IO(inputs[i, :], Fs)


filepath = 'D:\data\BramdW\TransientExp16apr2018'

if not os.path.exists(filepath):
        os.makedirs(filepath)
np.savez(os.path.join(filepath, 'inp11_5_15_outp1'), outputs = outputs)



#IVVIrack.setControlVoltages(ivvi, np.zeros(7))


