'''
Experiment description goes here.
'''
# Temporary workaround to incorporate skynet path into system path
import sys
import os 
current_dir = os.getcwd()
current_dir = current_dir.split(os.sep)
parent_dir = os.sep.join(current_dir[:-2])
sys.path.append(parent_dir)

# SkyNEt imports
import modules.SaveLib as SaveLib
from instruments.niDAQ import nidaqIO
import modules.Evolution as Evolution
from instruments.DAC import IVVIrack
import modules.PlotBuilder as PlotBuilder
import config_boolean_logic as config

# Other imports
import time
import numpy as np

#%% Initialization

# Initialize config object
config = config.experiment_config()

# Initialize input and target
t = config.InputGen()[0]  # Time array
x = np.asarray(config.InputGen()[1:3])  # Array with P and Q signal
w = config.InputGen()[3]  # Weight array
target = config.TargetGen()[1]  # Target signal

# np arrays to save genePools, outputs and fitness
geneArray = np.zeros((config.generations, config.genomes, config.genes))
outputArray = np.zeros((config.generations, config.genomes, len(x[0])))
fitnessArray = np.zeros((config.generations, config.genomes))

# Temporary arrays, overwritten each generation
fitnessTemp = np.zeros((config.genomes, config.fitnessavg))
outputAvg = np.zeros((config.fitnessavg, len(x[0])))
# outputTemp = np.zeros((config.genomes, len(x[0])))
controlVoltages = np.zeros(config.genes)

# Initialize save directory
saveDirectory = SaveLib.createSaveDirectory(config.filepath, config.name)

# Initialize main figure
mainFig = PlotBuilder.initMainFigEvolution(config.genes, config.generations, config.genelabels, config.generange)

# Initialize instruments
ivvi = IVVIrack.initInstrument()

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
        x_scaled = x * 0.5#genePool.MapGenes(config.generange[-1], genePool.pool[j, -1])

        ######### to be removed, noised added
        x_scaled[0,:] = x_scaled[0]+np.random.normal(0, 1, size=len(x_scaled[0,:]))*config.namp*j/config.genomes

        # Measure config.fitnessavg times the current configuration
        for avgIndex in range(config.fitnessavg):
            # Feed input to niDAQ
            output = nidaqIO.IO_2D(x_scaled, config.fs)

            # Plot genome
            PlotBuilder.currentGenomeEvolution(mainFig, genePool.pool[j])

            # Train output
            outputAvg[avgIndex] = config.amplification * np.asarray(output)  # empty for now, as we have only one output node

            # Calculate fitness
            fitnessTemp[j, avgIndex]= config.Fitness(outputAvg[avgIndex],
                                                     target,
                                                     w)

            # Plot output
            PlotBuilder.currentOutputEvolution(mainFig,
                                               t,
                                               target,
                                               output,
                                               j + 1, i + 1,
                                               fitnessTemp[j, avgIndex])

        # outputTemp[j] = outputAvg[np.argmin(fitnessTemp[j])]
        # saveDirectory = SaveLib.createSaveDirectory(config.filepath, config.name+'_genome_'+str(j))
        # SaveLib.saveMain(saveDirectory,
        #              geneArray,
        #              config.amplification * np.asarray(output),
        #              fitnessArray,
        #              t,
        #              x,
        #              config.amplification*target)

    genePool.fitness = fitnessTemp.min(1)  # Save fitness

    # Status print
    print("Generation nr. " + str(i + 1) + " completed")
    print("Highest fitness: " + str(max(genePool.fitness)))

    # Save generation data
    geneArray[i, :, :] = genePool.pool
#    outputArray[i, :, :] = outputTemp
    fitnessArray[i, :] = genePool.fitness

    # Update main figure
    PlotBuilder.updateMainFigEvolution(mainFig,
                                       geneArray,
                                       fitnessArray,
                                       outputArray,
                                       i + 1,
                                       t,
                                       config.amplification*target,
                                       output,
                                       w)

    #Save generation
    SaveLib.saveMain(saveDirectory,
                     geneArray,
                     outputArray,
                     fitnessArray,
                     t,
                     x,
                     config.amplification*target)

    # Evolve to the next generation
    genePool.NextGen()

PlotBuilder.finalMain(mainFig)
#raise KeyboardInterrupt
#
#finally:
inp = np.zeros((2,20))

controlVoltages = np.zeros(16)

IVVIrack.setControlVoltages(ivvi, controlVoltages)

   # feed 0 to nidaq
nidaqIO.IO_2D(inp, 1000)
#
fname = filepath + '\\main_figure.png'
plt.savefig(fname)
print('All good')

genePool = Evolution.GenePool(config.genes, config.genomes)
