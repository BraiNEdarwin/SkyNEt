'''
Experiment description goes here.
'''
<<<<<<< HEAD
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
=======

# SkyNEt imports
import SkyNEt.modules.SaveLib as SaveLib
from SkyNEt.instruments.niDAQ import nidaqIO
import SkyNEt.modules.Evolution as Evolution
from SkyNEt.instruments.DAC import IVVIrack
import SkyNEt.modules.PlotBuilder as PlotBuilder
>>>>>>> dev
import config_boolean_logic as config

# Other imports
import time
import numpy as np

#%% Initialization

# Initialize config object
<<<<<<< HEAD
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
outputTemp = np.zeros((config.genomes, len(x[0])))
controlVoltages = np.zeros(config.genes)

# Initialize save directory
saveDirectory = SaveLib.createSaveDirectory(config.filepath, config.name)

# Initialize main figure
mainFig = PlotBuilder.initMainFigEvolution(config.genes, config.generations, config.genelabels, config.generange)
=======
cf = config.experiment_config()

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
>>>>>>> dev

# Initialize instruments
ivvi = IVVIrack.initInstrument()

# Initialize genepool
<<<<<<< HEAD
genePool = Evolution.GenePool(config)

#%% Measurement loop

for i in range(config.generations):
    for j in range(config.genomes):
        # Set the DAC voltages
        for k in range(config.genes-1):
            controlVoltages[k] = genePool.MapGenes(
                                    config.generange[k], genePool.pool[j, k])
=======
genePool = Evolution.GenePool(cf)

#%% Measurement loop

for i in range(cf.generations):
    for j in range(cf.genomes):
        # Set the DAC voltages
        for k in range(cf.genes-1):
            controlVoltages[k] = genePool.MapGenes(
                                    cf.generange[k], genePool.pool[j, k])
>>>>>>> dev
        IVVIrack.setControlVoltages(ivvi, controlVoltages)
        time.sleep(1)  # Wait after setting DACs

        # Set the input scaling
<<<<<<< HEAD
        x_scaled = x * genePool.MapGenes(config.generange[-1], genePool.pool[j, -1])

        # Measure config.fitnessavg times the current configuration
        for avgIndex in range(config.fitnessavg):
            # Feed input to niDAQ
            output = nidaqIO.IO_2D(x_scaled, config.fs)
=======
        x_scaled = x * genePool.MapGenes(cf.generange[-1], genePool.pool[j, -1])

        # Measure cf.fitnessavg times the current configuration
        for avgIndex in range(cf.fitnessavg):
            # Feed input to niDAQ
            output = nidaqIO.IO_2D(x_scaled, cf.fs)
>>>>>>> dev

            # Plot genome
            PlotBuilder.currentGenomeEvolution(mainFig, genePool.pool[j])

            # Train output
<<<<<<< HEAD
            outputAvg[avgIndex] = config.amplification * np.asarray(output)  # empty for now, as we have only one output node

            # Calculate fitness
            fitnessTemp[j, avgIndex]= config.Fitness(outputAvg[avgIndex],
=======
            outputAvg[avgIndex] = cf.amplification * np.asarray(output)  # empty for now, as we have only one output node

            # Calculate fitness
            fitnessTemp[j, avgIndex]= cf.Fitness(outputAvg[avgIndex],
>>>>>>> dev
                                                     target,
                                                     w)

            # Plot output
            PlotBuilder.currentOutputEvolution(mainFig,
                                               t,
                                               target,
                                               output,
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
<<<<<<< HEAD
                                       config.amplification*target,
=======
                                       cf.amplification*target,
>>>>>>> dev
                                       output,
                                       w)

    # Save generation
    SaveLib.saveMain(saveDirectory,
                     geneArray,
                     outputArray,
                     fitnessArray,
                     t,
                     x,
<<<<<<< HEAD
                     config.amplification*target)
=======
                     cf.amplification*target)
>>>>>>> dev

    # Evolve to the next generation
    genePool.NextGen()

PlotBuilder.finalMain(mainFig)
#raise KeyboardInterrupt
#
#finally:
<<<<<<< HEAD
inp = np.zeros((2,20))

controlVoltages = np.zeros(16)

IVVIrack.setControlVoltages(ivvi, controlVoltages)

   # feed 0 to nidaq
nidaqIO.IO_2D(inp, config.fs)
=======
#    inp = np.zeros((2,20))
#
#    controlVoltages = np.zeros(16)
#
#    IVVIrack.setControlVoltages(ivvi, controlVoltages)
#
#    # feed 0 to nidaq
#    nidaqIO.IO_2D(inp, SampleFreq)
>>>>>>> dev
#
#    fname = filepath + '\\main_figure.png'
#    plt.savefig(fname)
#    print('All good')

<<<<<<< HEAD
# genePool = Evolution.GenePool(config.genes, config.genomes)
=======
# genePool = Evolution.GenePool(cf.genes, cf.genomes)
>>>>>>> dev
