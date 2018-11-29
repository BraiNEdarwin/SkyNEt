#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 16:22:10 2018

@author: ljknoll
"""
import sys
sys.path
sys.path.append('../../../')
# SkyNEt imports
import SkyNEt.modules.SaveLib as SaveLib
import SkyNEt.modules.Evolution as Evolution
import SkyNEt.modules.PlotBuilder as PlotBuilder
import SkyNEt.experiments.boolean_logic.config_evoNN_template as config
from SkyNEt.modules.Nets.staNNet import staNNet 
# Other imports
import torch
import time
import numpy as np

# Initialization

# Initialize config object
cf = config.experiment_config()
# generations; the amount of generations for the GA
cf.generations = 500
# generange; the range that each gene ([0, 1]) is mapped to. E.g. in the Boolean
# experiment the genes for the control voltages are mapped to the desired
# control voltage range.
cf.generange = [[-600,600], [-900, 900], [-900, 900], [-900, 900], [-600, 600], [0.1, 0.5]]
cf.default_partition = [5, 5, 5, 5, 5]
cf.partition = cf.default_partition.copy()
# genomes; the amount of genomes in the genepool
cf.genomes = 25
# genes; the amount of genes per genome
cf.genes = 6
# genes; the amount of genes per genome
cf.mutationrate = 0.1
# fitnessavg; the amount of times the same genome is tested to obtain the fitness value.
cf.fitnessavg = 1
# fitnessparameters; the parameters for FitnessEvolution
cf.fitnessparameters = [1, 0, 1, 0.01]

# Initialize input and target
t = cf.InputGen()[0]  # Time array
x = np.asarray(cf.InputGen()[1:3])  # Array with P and Q signal
w = cf.InputGen()[3]  # Weight array
target = cf.TargetGen()[1]  # Target signal

# np arrays to save genePools, outputs and fitness
geneArray = np.zeros((cf.generations, cf.genomes, cf.genes))
outputArray = np.zeros((cf.generations, cf.genomes, len(x[0])))
fitnessArray = np.zeros((cf.generations, cf.genomes))

# Temporary arrays, overwritten each generation
fitnessTemp = np.zeros((cf.genomes, cf.fitnessavg))
outputAvg = np.zeros((cf.fitnessavg, len(x[0])))
outputTemp = np.zeros((cf.genomes, len(x[0])))
controlVoltages = np.zeros(cf.genes)

# Initialize save directory
saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# Initialize main figure
mainFig = PlotBuilder.initMainFigEvolution(cf.genes, cf.generations, cf.genelabels, cf.generange)

# Initialize NN
main_dir = r'/home/lennart/Desktop/nnweb/'
data_dir = 'lr2e-4_eps400_mb512_20180807CP.pt'
net = staNNet(main_dir+data_dir)

# Initialize genepool
genePool = Evolution.GenePool(cf)

# Measurement loop

for i in range(cf.generations):
    start = time.time()
    for j in range(cf.genomes):
        # Set the DAC voltages
        for k in range(cf.genes-1):
            controlVoltages[k] = genePool.MapGenes(
                                    cf.generange[k], genePool.pool[j, k])

        # Set the input scaling
        x_scaled = x * genePool.config_obj.input_scaling

        # Measure cf.fitnessavg times the current configuration
        for avgIndex in range(cf.fitnessavg):
            # Feed input to NN
            g = np.ones_like(target)[:,np.newaxis]*genePool.pool[j][:,np.newaxis].T
            x_dummy = np.concatenate((x_scaled.T,g),axis=1) # First input then genes; dims of input TxD
            inputs = torch.from_numpy(x_dummy).type(dtype)
            inputs = Variable(inputs)
            output = net.outputs(inputs)
 
#            # Plot genome
#            PlotBuilder.currentGenomeEvolution(mainFig, genePool.pool[j])

            # Train output
            outputAvg[avgIndex] = cf.amplification * np.asarray(output)  # empty for now, as we have only one output node

            # Calculate fitness
            fitnessTemp[j, avgIndex]= cf.Fitness(outputAvg[avgIndex],
                                                     target,
                                                     w)

#            # Plot output
#            PlotBuilder.currentOutputEvolution(mainFig,
#                                               t,
#                                               target,
#                                               output,
#                                               j + 1, i + 1,
#                                               fitnessTemp[j, avgIndex])

        outputTemp[j] = outputAvg[np.argmin(fitnessTemp[j])]

    genePool.fitness = fitnessTemp.min(1)  # Save fitness
    end = time.time()
    # Status print
    print("Generation nr. " + str(i + 1) + " completed; took "+str(end-start)+" sec.")
    print("Highest fitness: " + str(max(genePool.fitness)))

    # Save generation data
    geneArray[i, :, :] = genePool.pool
    outputArray[i, :, :] = outputTemp
    fitnessArray[i, :] = genePool.fitness

#    # Update main figure
#    PlotBuilder.updateMainFigEvolution(mainFig,
#                                       geneArray,
#                                       fitnessArray,
#                                       outputArray,
#                                       i + 1,
#                                       t,
#                                       cf.amplification*target,
#                                       output,
#                                       w)

    # Save generation
    SaveLib.saveMain(saveDirectory,
                     geneArray,
                     outputArray,
                     fitnessArray,
                     t,
                     x,
                     cf.amplification*target)

    # Evolve to the next generation
    genePool.NextGen()

PlotBuilder.finalMain(mainFig)
