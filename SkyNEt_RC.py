''''
Measurement script to perform a simple RC experiment.
'''
# Import packages
import modules.ReservoirFull as Reservoir
import modules.PlotBuilder as PlotBuilder
import modules.GenerateInput as GenerateInput
import modules.Evolution as Evolution
import modules.PostProcess as PostProcess
import modules.SaveLib as SaveLib
from instruments.ADwin import adwinIO
from instruments.DAC import IVVIrack

# temporary imports
import numpy as np


def mapGenes(generange, gene):
    return generange[0] + gene * (generange[1] - generange[0])


# Read config.txt file
exec(open("config.txt").read())

# initialize genepool
genePool = Evolution.GenePool(genes, genomes)

# initialize benchmark
# Obtain benchmark input
[t, inp] = GenerateInput.softwareInput(
    benchmark, SampleFreq, WavePeriods, WaveFrequency)
# Obtain benchmark output
[t, outp] = GenerateInput.targetOutput(
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

# initialize main figure
mainFig = PlotBuilder.initMainFig(genes, generations, genelabels, generange)

# initialize instruments
adw = adwinIO.initInstrument()
ivvi = IVVIrack.initInstrument()

for i in range(generations):

    for j in range(genomes):

        # set the DAC voltages
        for k in range(genes):
            controlVoltages[k] = mapGenes(generange[k], genePool.pool[k, j])
        IVVIrack.setControlVoltages(ivvi, controlVoltages)

        for avgIndex in range(fitnessAvg):

            # feed input to adwin
            output = adwinIO.IO(adw, inp, SampleFreq)

            # Train output
            trained_output = output  # empty for now, as we have only one output node

            fitnessTemp[j, avgIndex] = PostProcess.fitness(
                trained_output[:, avgIndex], outp[skipstates:])

        outputTemp[:, j] = trained_output[:, np.argmin(fitnessTemp[j, :])]

    genePool.fitness = fitnessTemp.min(1)
    print("Generation nr. " + str(i + 1) + " completed")
    print("Highest fitness: " + str(max(genePool.fitness)))

    # save generation data
    geneArray[i, :, :] = genePool.returnPool()
    outputArray[i, :, :] = outputTemp
    fitnessArray[i, :] = fitnessTemp.min(1)

    PlotBuilder.updateMainFig(mainFig, geneArray, fitnessArray,
                              outputArray, i + 1, t[skipstates:], outp[skipstates:])

    # evolve the next generation
    genePool.nextGen()

SaveLib.saveMain(filepath, geneArray, outputArray, fitnessArray, t, inp, outp)

PlotBuilder.finalMain(mainFig)
