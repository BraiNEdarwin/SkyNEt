# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 11:32:22 2018
Find the correct control voltages for sampling on various outputs.
Code is based on the boolean logic finder.

@author: Mark
"""

# SkyNEt imports
import modules.SaveLib as SaveLib
from instruments.niDAQ import nidaqIO
import modules.Evolution as Evolution
from instruments.DAC import IVVIrack
import modules.PlotBuilder as PlotBuilder

# Other imports
import time
import numpy as np

#%% Initialization


def CVFinder(config, outputTarget, instrumentInit):
    # Initialize input and target
    t = config.InputGen()[0]  # Time array
    x = np.asarray(config.InputGen()[1:3])  # Array with P and Q signal
    w = config.InputGen()[3]  # Weight array
    target = outputTarget  # Target signal
    ivvi = instrumentInit
    
    # np arrays to save genePools, outputs and fitness
    geneArray = np.zeros((config.generations, config.genomes, config.genes))
    outputArray = np.zeros((config.generations, config.genomes, len(x[0])))
    fitnessArray = np.zeros((config.generations, config.genomes))
    
    # Temporary arrays, overwritten each generation
    fitnessTemp = np.zeros((config.genomes, config.fitnessavg))
    outputAvg = np.zeros((config.fitnessavg, len(x[0])))
    outputTemp = np.zeros((config.genomes, len(x[0])))
    controlVoltages = np.zeros(config.genes)
    
    # Initialize save directory
    saveDirectory = SaveLib.createSaveDirectory(config.filepath, config.nameCV + '_target_' + str(outputTarget[0])[:4].replace('.','_'))
    
    # Initialize main figure
    mainFig = PlotBuilder.initMainFigEvolution(config.genes, config.generations, config.genelabels, config.generange)
    
    # Initialize genepool
    genePool = Evolution.GenePool(config)
    
    #%% Measurement loop
    
    for i in range(config.generations):
        for j in range(config.genomes):
            # Set the DAC voltages
            for k in range(config.genes-1):
                controlVoltages[k] = genePool.MapGenes(
                                        config.generange[k], genePool.pool[j, k])
            IVVIrack.setControlVoltages(ivvi, controlVoltages)
            time.sleep(1)  # Wait after setting DACs
    
            # Set the input scaling
            x_scaled = x * genePool.MapGenes(config.generange[-1], genePool.pool[j, -1])
    
            # Measure config.fitnessavg times the current configuration
            for avgIndex in range(config.fitnessavg):
                # Feed input to niDAQ
                output = nidaqIO.IO_2D(x_scaled, config.fs)
    
                # Plot genome
                PlotBuilder.currentGenomeEvolution(mainFig, genePool.pool[j])
    
                # Train output
                outputAvg[avgIndex] = config.amplification * np.asarray(output) / config.postgain  # empty for now, as we have only one output node
    
                # Calculate fitness
                fitnessTemp[j, avgIndex]= config.Fitness(outputAvg[avgIndex],
                                                         target)
    
                # Plot output
                PlotBuilder.currentOutputEvolution(mainFig,
                                                   t,
                                                   target,
                                                   config.amplification/config.postgain*np.array(output),
                                                   j + 1, i + 1,
                                                   fitnessTemp[j, avgIndex])
    
            outputTemp[j] = outputAvg[np.argmin(fitnessTemp[j])]
    
        genePool.fitness = fitnessTemp.min(1)  # Save fitness
        
        # Status print
        print("Generation nr. " + str(i + 1) + " completed")
        print("Highest fitness: " + str(max(genePool.fitness)))
    
        # Save generation data
        geneArray[i, :, :] = genePool.pool
        outputArray[i, :, :] = outputTemp
        fitnessArray[i, :] = genePool.fitness
    
        # Update main figure
        PlotBuilder.updateMainFigEvolution(mainFig,
                                           geneArray,
                                           fitnessArray,
                                           outputArray,
                                           i + 1,
                                           t,
                                           target,
                                           config.amplification/config.postgain*np.array(output),
                                           w)
    
        # Save generation
        SaveLib.saveExperiment(config.configSrc, saveDirectory,
                     geneArray = geneArray,
                     outputArray = outputArray,
                     fitnessArray = fitnessArray,
                     t = t,
                     x = x,
                     amplified_target = config.amplification/config.postgain*target)
        
        
        if max(genePool.fitness) > config.fitThres:
            bestGenes = geneArray[i, genePool.fitness.argmax(), :]
            bestCV = np.zeros(len(bestGenes) - 1)
            for k in range(len(bestGenes) - 1):
                bestCV[k] = genePool.MapGenes(config.generange[k], bestGenes[k])        
            return bestCV       # If fitness is high enough, break the GA and keep this value
        # Evolve to the next generation
        genePool.NextGen()
    
    # If threshold fitness is not reached, take best CV found so far
    bestGeneration = int(fitnessArray.argmax() / config.genomes)
    bestGenome = int(fitnessArray.argmax() % config.genomes)
    bestGenes = geneArray[bestGeneration, bestGenome , :]
    bestCV = np.zeros(len(bestGenes) - 1)
    for k in range(len(bestGenes) - 1):
        bestCV[k] = genePool.MapGenes(config.generange[k], bestGenes[k])
    return bestCV

    #raise KeyboardInterrupt
    #
    #finally:
    #    inp = np.zeros((2,20))
    #
    #    controlVoltages = np.zeros(16)
    #
    #    IVVIrack.setControlVoltages(ivvi, controlVoltages)
    #
    #    # feed 0 to nidaq
    #    nidaqIO.IO_2D(inp, SampleFreq)
    #
    #    fname = filepath + '\\main_figure.png'
    #    plt.savefig(fname)
    #    print('All good')
    
    # genePool = Evolution.GenePool(config.genes, config.genomes)
