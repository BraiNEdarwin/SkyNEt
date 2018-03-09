''''
Main handler of the SkyNEt platform
'''
# Import packages
import modules.ReservoirFull as Reservoir
import modules.PlotBuilder as PlotBuilder
import modules.GenerateInput as GenerateInput
import modules.Evolution as Evolution
import modules.PostProcess as PostProcess
import math
# temporary imports
import numpy as np


def mapGenes(generange, gene):
    return generange[0] + gene * (generange[1] - generange[0])


# Read config.txt file
exec(open("config.txt").read())

# initialize genepool
genePool = Evolution.GenePool(genes, genomes)
fitnessArray = np.empty(genomes)

#initialize benchmark
# Obtain benchmark input
[t, inp] = GenerateInput.softwareInput(
    benchmark, SampleFreq, WavePeriods, WaveFrequency)
# Obtain benchmark output
[t, outp] = GenerateInput.targetOutput(
    benchmark, SampleFreq, WavePeriods, WaveFrequency)

#np arrays to save genePools, outputs and fitness
geneArray = np.empty((generations, genes, genomes))
outputArray = np.empty((generations, len(inp) - skipstates, genomes))
fitnessArray = np.empty((generations, genomes))

#temporary arrays, overwritten each generation
fitnessTemp = np.empty(genomes)
outputTemp = np.empty((len(inp) - skipstates, genomes))


for i in range(generations):

    for j in range(genomes):

        nodes[1] = mapGenes(generange[0], genePool.pool[0, j])
        nodes[1] = int(nodes[1])
        inputscaling = mapGenes(generange[1], genePool.pool[1, j])
        spectralradius = mapGenes(generange[2], genePool.pool[2, j])
        weightdensity = mapGenes(generange[3], genePool.pool[3, j])

        # Init software reservoir
        res = Reservoir.Network(nodes, inputscaling,
                                spectralradius, weightdensity)

        for k in range(len(inp)):
            res.update_reservoir(inp[k])

        trained_output = res.train_reservoir_ridgereg(
            outp, rralpha, skipstates)
        # trained_output = res.train_reservoir_pseudoinv(outp, skipstates)
        fitnessTemp[j] = PostProcess.fitness(trained_output, outp[skipstates:])
        outputTemp[:,j] = trained_output
      
            
    genePool.fitness = fitnessTemp
    print("Generation nr. " + str(i + 1) + " completed")
    print("Highest fitness: " + str(max(genePool.fitness)))

    #save generation data
    geneArray[i, :, :] = genePool.returnPool()
    outputArray[i, :, :] = outputTemp
    fitnessArray[i, :] = fitnessTemp

    #evolve the next generation
    genePool.nextGen()
    
PlotBuilder.bigDaddy(geneArray, fitnessArray)

#PlotBuilder.genericPlot1D(t[skipstates:], trained_output, 'time', 'y', 'test')
#PlotBuilder.genericPlot1D(t[skipstates:], outp[skipstates:], 'time', 'y', 'test')
#PlotBuilder.showPlot()