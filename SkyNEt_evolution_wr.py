''''
Measurement script to perform an evolution experiment of a selected
gate. This will initially be tested on the Heliox (with nidaq) setup.
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

# temporary imports
import numpy as np


# Read config.txt file
exec(open("config.txt").read())

# initialize genepool
genePool = Evolution.GenePool(genes, genomes)

# initialize benchmark
# Obtain benchmark input (P and Q are input1, input2)
[t, inp1] = GenerateInput.softwareInput(benchmark, SampleFreq, WavePeriods, WaveFrequency)

WavePeriods2 = (WavePeriods/WaveFrequency)*WaveFrequency2

[t, inp2] = GenerateInput.softwareInput(benchmark, SampleFreq, WavePeriods2, WaveFrequency2)

inp = np.empty((2,len(inp2)))
inp[0,:]=inp1*2
inp[1,:]=inp2*2

# format for nidaq
#x = np.empty((2, len(P)))

# Obtain benchmark target
[t, target] = GenerateInput.targetOutput(
    benchmark, SampleFreq, WavePeriods, WaveFrequency)

# np arrays to save genePools, outputs and fitness
geneArray = np.empty((generations, genes, genomes))
outputArray = np.empty((generations, len(inp) - skipstates, genomes))
fitnessArray = np.empty((generations, genomes))

# temporary arrays, overwritten each generation
fitnessTemp = np.empty((genomes, fitnessAvg))
trained_output = np.empty((len(inp) - skipstates, fitnessAvg))
outputTemp = np.empty((len(inp) - skipstates, genomes))
controlVoltages = np.empty(genes)

# initialize save directory
saveDirectory = SaveLib.createSaveDirectory(filepath, name)

# initialize main figure
mainFig = PlotBuilder.initMainFigEvolution(genes, generations, genelabels, generange)


# initialize instruments
ivvi = IVVIrack.initInstrument()

for i in range(generations):

    for j in range(genomes):

        # set the DAC voltages
        for k in range(genes-1):
            controlVoltages[k] = Evolution.mapGenes(
                generange[k], genePool.pool[k, j])
        IVVIrack.setControlVoltages(ivvi, controlVoltages)

        #set the input scaling
        #x_scaled = x * Evolution.mapGenes(generange[-1], genePool.pool[genes-1, j])

        #wait after setting DACs
        time.sleep(1)

        for avgIndex in range(fitnessAvg):

            # feed input to adwin
            output = nidaqIO.IO_2D(inp, SampleFreq)

            # plot genome
            PlotBuilder.currentGenomeEvolution(mainFig, genePool.pool[:, j])
            
            # Train output
            trained_output[:, avgIndex] = output[0]  # empty for now, as we have only one output node

            # Calculate fitness
            fitnessTemp[j, avgIndex] = PostProcess.fitness(trained_output[:, avgIndex], target[skipstates:][1])

            #plot output
            PlotBuilder.currentOutputEvolution(mainFig, t, target, output, j + 1, i + 1, fitnessTemp[j, avgIndex])

        outputTemp[:, j] = trained_output[:, np.argmin(fitnessTemp[j, :])]

    genePool.fitness = fitnessTemp.min(1)
    print("Generation nr. " + str(i + 1) + " completed")
    print("Highest fitness: " + str(max(genePool.fitness)))

    # save generation data
    geneArray[i, :, :] = genePool.returnPool()
    outputArray[i, :, :] = outputTemp
    fitnessArray[i, :] = fitnessTemp.min(1)

    PlotBuilder.currentOutputEvolution(mainFig, t, target, output, j + 1, i + 1, fitnessTemp[j, avgIndex])
    PlotBuilder.updateMainFigEvolution(mainFig, geneArray, fitnessArray, outputArray, i + 1, t, target, output)
	
	#save generation
    SaveLib.saveMain(saveDirectory, geneArray, outputArray, fitnessArray, t, inp, target)
	
    # evolve the next generation
    genePool.nextGen()

SaveLib.saveMain(filepath, geneArray, outputArray, fitnessArray, t, inp, target)

PlotBuilder.finalMain(mainFig)
