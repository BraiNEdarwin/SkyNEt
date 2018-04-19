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

# temporary imports
import numpy as np

filepath = ''
name = ''

voltageGrid = [-2000, -1600, -1200, -800, -400, 0, 400, 800, 1600, 2000]

controlVoltages = np.empty(genes)

# initialize save directory
saveDirectory = SaveLib.createSaveDirectory(filepath, name)


# initialize instruments
ivvi = IVVIrack.initInstrument()

for i in range(generations):

    for j in range(genomes):

        # set the DAC voltages
        for k in range(genes):
            controlVoltages[k] = Evolution.mapGenes(
                generange[k], genePool.pool[k, j])
        IVVIrack.setControlVoltages(ivvi, controlVoltages)

        for avgIndex in range(fitnessAvg):

            # feed input to adwin
            output = nidaqIO.IO_2D(x, SampleFreq)

            # plot genome
            PlotBuilder.currentGenomeEvolution(mainFig, genePool.pool[:, j])
            
            # Train output
            trained_output[:, avgIndex] = output  # empty for now, as we have only one output node

            # Calculate fitness
            fitnessTemp[j, avgIndex] = PostProcess.fitnessEvolution(
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

    PlotBuilder.updateMainFigEvolution(mainFig, geneArray, fitnessArray, outputArray, i + 1, t, target, output)
	
	#save generation
    SaveLib.saveMain(saveDirectory, geneArray, outputArray, fitnessArray, t, x, target)
	
    # evolve the next generation
    genePool.nextGen()

SaveLib.saveMain(filepath, geneArray, outputArray, fitnessArray, t, x, target)

PlotBuilder.finalMain(mainFig)
