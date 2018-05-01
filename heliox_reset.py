''''
Reset NI outputs to zero and IVVI DACs to zero
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

# temporary imports
import numpy as np


def mapGenes(generange, gene):
    return generange[0] + gene * (generange[1] - generange[0])


# Read config.txt file
exec(open("config.txt").read())


# initialize benchmark
inp = np.zeros((2,20))


# initialize instruments
ivvi_reset = IVVIrack.initInstrument()

controlVoltages = np.zeros(16)

# set the DAC voltages
IVVIrack.setControlVoltages(ivvi_reset, controlVoltages)

# feed 0 to nidaq
nidaqIO.IO_2D(inp, SampleFreq)

print('All good')