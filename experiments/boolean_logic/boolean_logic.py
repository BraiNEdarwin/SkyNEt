'''
In this script the controlvoltages corresponding to different genomes are set and a boulean logic measurement is performed.
This is done withing 3 for loops:
1. for i in range generations for every generation rankes the fitnsessed of the genomes (control voltage sets) of the generation and used the GA to define the genomes of the next generation.
   After this the output with the best fitness as well as this fitness compared to the generation are plotted.
2. for j in range genomes for ever genome in a generation it sets the desisred control voltages on the DAC's of the IVVI-rack and scales the boulean logic input.
   Next, it uses either the adwin or nidaq to measure the boulean logic input output relation after which the coresponding fitness is calculated.
   Finally, the genomes and output of each genome is plotted.
3. for k in range genes is use to map the 0-1 generated genome to the generange at put it in the desired orded in the coltrolvoltage array. 


'''

# SkyNEt imports
import SkyNEt.modules.SaveLib as SaveLib
from SkyNEt.instruments.niDAQ import nidaqIO
import SkyNEt.modules.Evolution as Evolution
from SkyNEt.instruments.DAC import IVVIrack
import SkyNEt.modules.PlotBuilder as PlotBuilder
import config_boolean_logic as config

# Other imports
import time
import numpy as np

#%% Initialization

# Initialize config object
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

# Initialize instruments
ivvi = IVVIrack.initInstrument()

# Initialize genepool
genePool = Evolution.GenePool(cf)

#%% Measurement loop

for i in range(cf.generations):
    for j in range(cf.genomes):
        # Set the DAC voltages
        for k in range(cf.genes-1):
            controlVoltages[k] = genePool.MapGenes(
                                    cf.generange[k], genePool.pool[j, k])
        IVVIrack.setControlVoltages(ivvi, controlVoltages)
        time.sleep(1)  # Wait after setting DACs

        # Set the input scaling
        x_scaled = x * genePool.MapGenes(cf.generange[-1], genePool.pool[j, -1])

        # Measure cf.fitnessavg times the current configuration
        for avgIndex in range(cf.fitnessavg):
            # Feed input to niDAQ
            output = nidaqIO.IO_2D(x_scaled, cf.fs)

            # Plot genome
            PlotBuilder.currentGenomeEvolution(mainFig, genePool.pool[j])

            # Train output
            outputAvg[avgIndex] = cf.amplification * np.asarray(output)  # empty for now, as we have only one output node

            # Calculate fitness
            fitnessTemp[j, avgIndex]= cf.Fitness(outputAvg[avgIndex],
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
                                       cf.amplification*target,
                                       output,
                                       w)

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

# genePool = Evolution.GenePool(cf.genes, cf.genomes)
