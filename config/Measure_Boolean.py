import modules.SaveLib as SaveLib
from instruments.niDAQ import nidaqIO
import modules.Evolution as Evolution
from instruments.DAC import IVVIrack
import modules.PlotBuilder as PlotBuilder
import config_Boolean as config
import time

# temporary imports
import numpy as np

#make it so that the data from the config is imported such that it can be called using config.X where x is your variable.
config = config.experiment_config()
#
#InputGen = config.InputGen()
#plt.plot(config.InputGen()[0], config.InputGen()[3])

# np arrays to save genePools, outputs and fitness
geneArray = np.zeros((config.generations(), config.genes(), config.genomes()))
outputArray = np.zeros((config.generations(), config.InputGen()[1] - config.skipstates(), config.genomes()))
fitnessArray = np.zeros((config.generations(), config.genomes()))

# temporary arrays, overwritten each generation
fitnessTemp = np.zeros((config.genomes(), config.fitnessAvg()))
trained_output = np.zeros((config.InputGen()[1] - config.skipstates(), config.fitnessAvg()))
outputTemp = np.zeros((config.InputGen()[1] - config.skipstates(), conifg.genomes()))
controlVoltages = np.zeros(config.genes())

# initialize save directory
saveDirectory = SaveLib.createSaveDirectory(config.filepath(), config.name())

# initialize main figure
mainFig = PlotBuilder.initMainFigEvolution(config.genes(), config.generations(), config.genelabels(), config.generange())

# initialize instruments
ivvi = IVVIrack.initInstrument()

# initialize genepool
genePool = Evolution.GenePool(config.genes(), config.genomes())

for i in range(config.generations()):
    for j in range(config.genomes()):

        # set the DAC voltages
        for k in range(config.genes()-1):
            controlVoltages[k] = Evolution.mapGenes(
                config.generange()[k], genePool.pool[k, j])
        IVVIrack.setControlVoltages(ivvi, controlVoltages)

        #set the input scaling
        x_scaled = config.InputGen()[1:2] * Evolution.mapGenes(config.generange()[-1], genePool.pool[config.genes()-1, j])

        #wait after setting DACs
        time.sleep(1)

        for avgIndex in range(config.fitnessAvg()):

            # feed input to adwin
            output = nidaqIO.IO_2D(x_scaled, config.fs())

            # plot genome
            PlotBuilder.currentGenomeEvolution(mainFig, genePool.pool[:, j])
                
            # Train output
            trained_output[:, avgIndex] = config.amplification() * np.asarray(output)  # empty for now, as we have only one output node

            # Calculate fitness
            fitnessTemp[j, avgIndex]= config.Fitness(
                trained_output[:, avgIndex], config.TargetGen()[config.skipstates():], config.InputGen()[3])

            #plot output
            PlotBuilder.currentOutputEvolution(mainFig, config.InputGen()[0], config.TargetGen()[1], output, j + 1, i + 1, fitnessTemp[j, avgIndex])

        outputTemp[:, j] = trained_output[:, np.argmin(fitnessTemp[j, :])]

    genePool.fitness = fitnessTemp.min(1)
    print("Generation nr. " + str(i + 1) + " completed")
    print("Highest fitness: " + str(max(genePool.fitness)))

    # save generation data
    geneArray[i, :, :] = genePool.returnPool()
    outputArray[i, :, :] = outputTemp
    fitnessArray[i, :] = fitnessTemp.min(1)

    PlotBuilder.currentOutputEvolution(mainFig, config.InputGen()[0], config.amplification() *config.TargetGen()[1], output, j + 1, i + 1, fitnessTemp[j, avgIndex])
    PlotBuilder.updateMainFigEvolution(mainFig, geneArray, fitnessArray, outputArray, i + 1, config.InputGen()[0], config.amplification() *config.TargetGen()[1], output, config.InputGen()[3])
    	
    	#save generation
    SaveLib.saveMain(saveDirectory, geneArray, outputArray, fitnessArray, config.InputGen()[0], config.InputGen()[1:2], config.amplification() *config.TargetGen()[1])
    	
    # evolve the next generation
    genePool.nextGen()

SaveLib.saveMain(config.filepath(), geneArray, outputArray, fitnessArray, config.InputGen()[0], config.InputGen()[1:2], config.amplification() *config.TargetGen()[1])

PlotBuilder.finalMain(mainFig)
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