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

# print (list(iter.product(a, repeat=6)))

# Read config.txt file
exec(open("config.txt").read())

# initialize genepool
genePool = Evolution.GenePool(genes, genomes)

# initialize benchmark
# Obtain benchmark input (P and Q are input1, input2)
[x_spiral1, y_spiral1, x_spiral2, y_spiral2] = GenerateInput.SpiralInput(n_points, SpiralOffset)

# format for nidaq
x = np.empty((2, len(x_spiral1)))
x[0,:] = x_spiral1 * 800
x[1,:] = y_spiral1 * 800


t = np.linspace(0, len(x_spiral1)/samplefreq ,len(x_spiral1))
w = np.ones(len(x_spiral1))


# Obtain benchmark target
target = t

# np arrays to save genePools, outputs and fitness
geneArray = np.empty((generations, genes, genomes))
outputArray = np.empty((generations, len(P) - skipstates, genomes))
fitnessArray = np.empty((generations, genomes))

# temporary arrays, overwritten each generation
fitnessTemp = np.empty((genomes, fitnessAvg))
trained_output = np.empty((len(P) - skipstates, fitnessAvg))
outputTemp = np.empty((len(P) - skipstates, genomes))
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
        x_scaled = x * Evolution.mapGenes(generange[-1], genePool.pool[genes-1, j])

        #wait after setting DACs
        time.sleep(1)

        for avgIndex in range(fitnessAvg):

            # feed input to adwin
            output = nidaqIO.IO_2D(x_scaled, SampleFreq)

            # plot genome
            PlotBuilder.currentGenomeEvolution(mainFig, genePool.pool[:, j])
            
            # Train output
            trained_output[:, avgIndex] =10 * np.asarray(output)  # empty for now, as we have only one output node

            # Calculate fitness
            fitnessTemp[j, avgIndex]= PostProcess.fitnessEvolutionSpiral(
                trained_output[:, avgIndex], target[skipstates:], W, fitnessParameters)

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
    SaveLib.saveMain(saveDirectory, geneArray, outputArray, fitnessArray, t, x, target)
	
    # evolve the next generation
    genePool.nextGen()

SaveLib.saveMain(filepath, geneArray, outputArray, fitnessArray, t, x, target)

PlotBuilder.finalMain(mainFig)
