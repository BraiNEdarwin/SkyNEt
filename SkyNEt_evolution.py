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
try:
    # Read config.txt file
    exec(open("config.txt").read())

    # initialize genepool
    genePool = Evolution.GenePool(genes, genomes)

    # initialize benchmark
    # Obtain benchmark input (P and Q are input1, input2)
    [t, P, Q, W] = GenerateInput.softwareInput(benchmark, SampleFreq, WavePeriods, WaveFrequency)

    # format for nidaq
    x = np.empty((2, len(P)))
    x[0,:] = P
    x[1,:] = Q
    # Obtain benchmark target
    [t, target] = GenerateInput.targetOutput(
        benchmark, SampleFreq, WavePeriods, WaveFrequency)

    # np arrays to save genePools, outputs and fitness
    geneArray = np.zeros((generations, genes, genomes))
    outputArray = np.zeros((generations, len(P) - skipstates, genomes))
    fitnessArray = np.zeros((generations, genomes))

    # temporary arrays, overwritten each generation
    fitnessTemp = np.zeros((genomes, fitnessAvg))
    trained_output = np.zeros((len(P) - skipstates, fitnessAvg))
    outputTemp = np.zeros((len(P) - skipstates, genomes))
    controlVoltages = np.zeros(genes)

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
                fitnessTemp[j, avgIndex]= PostProcess.fitnessEvolution(
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
        PlotBuilder.updateMainFigEvolution(mainFig, geneArray, fitnessArray, outputArray, i + 1, t, 10*target, output, W)
    	
    	#save generation
        SaveLib.saveMain(saveDirectory, geneArray, outputArray, fitnessArray, t, x, target)
    	
        # evolve the next generation
        genePool.nextGen()

    SaveLib.saveMain(filepath, geneArray, outputArray, fitnessArray, t, x, target)

    PlotBuilder.finalMain(mainFig)
    raise KeyboardInterrupt

finally:
    inp = np.zeros((2,20))

    controlVoltages = np.zeros(16)

    IVVIrack.setControlVoltages(ivvi, controlVoltages)

    # feed 0 to nidaq
    nidaqIO.IO_2D(inp, SampleFreq)

    fname = filepath + '\\main_figure.png'
    plt.savefig(fname)
    print('All good')