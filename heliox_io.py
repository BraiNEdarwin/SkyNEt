''''
Primitive measurement script to test i/o with the heliox setup
and NI usb-6216
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
Obtain benchmark input
[t, inp] = GenerateInput.softwareInput(
    benchmark, SampleFreq, WavePeriods, WaveFrequency)
# Obtain benchmark output
[t, outp] = GenerateInput.targetOutput(
    benchmark, SampleFreq, WavePeriods, WaveFrequency)


# temporary arrays, overwritten each generation
fitnessTemp = np.empty((genomes, fitnessAvg))
trained_output = np.empty((len(inp) - skipstates, fitnessAvg))
outputTemp = np.empty((len(inp) - skipstates, genomes))
controlVoltages = np.random.rand(genes)


# initialize instruments
ivvi = IVVIrack.initInstrument()


# set the DAC voltages
for k in range(genes):
	IVVIrack.setControlVoltages(ivvi, controlVoltages)


# feed input to adwin
output = nidaqIO.IO(inp, SampleFreq)


PlotBuilder.genericPlot1D(t, output, 'time', 'voltage', 'heliox io test')
PlotBuilder.showPlot()


