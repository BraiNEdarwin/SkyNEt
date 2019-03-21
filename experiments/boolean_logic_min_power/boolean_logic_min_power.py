'''
In this script, a regular Boolean logic search is performed. 
Additionally, the currents through all electrodes are recorded with
a resistor box. 
The fitness function includes a term that tries to minimize total
power.
'''

# SkyNEt imports
import SkyNEt.modules.SaveLib as SaveLib
import SkyNEt.modules.Evolution as Evolution
import SkyNEt.modules.PlotBuilder as PlotBuilder
import config_boolean_logic_min_power as config
from SkyNEt.instruments import InstrumentImporter

# Other imports
import time
import numpy as np

# Initialize config object
cf = config.experiment_config()

# Initialize input and target
t = cf.InputGen()[0]  # Time array
x = np.asarray(cf.InputGen()[1:3])  # Array with P and Q signal
w = cf.InputGen()[3]  # Weight array
target = cf.TargetGen()[1]  # Target signal

# np arrays to save genePools, outputs and fitness
geneArray = np.zeros((cf.generations, cf.genomes, cf.genes))
outputArray = np.zeros((cf.generations, cf.genomes, 8, len(x[0])))
fitnessArray = np.zeros((cf.generations, cf.genomes))

# Temporary arrays, overwritten each generation
fitnessTemp = np.zeros((cf.genomes, cf.fitnessavg))
outputAvg = np.zeros((cf.fitnessavg, 8, len(x[0])))
outputTemp = np.zeros((cf.genomes, 8, len(x[0])))
inputTemp = np.zeros((cf.genomes, 8, len(x[0])))
controlVoltages = np.zeros(cf.genes)

# Initialize save directory
saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# Initialize main figure
mainFig = PlotBuilder.initMainFigEvolution(cf.genes, cf.generations, cf.genelabels, cf.generange)

# Initialize instruments
ivvi = InstrumentImporter.IVVIrack.initInstrument()

# Initialize genepool
genePool = Evolution.GenePool(cf)

# Initialize V array
V = np.zeros((8, len(x[0])))

#%% Measurement loop

for i in range(cf.generations):
    for j in range(cf.genomes):
        # Set the DAC voltages
        for k in range(cf.genes-1):
            controlVoltages[k] = genePool.MapGenes(
                                    cf.generange[k], genePool.pool[j, k])
        InstrumentImporter.IVVIrack.setControlVoltages(ivvi, controlVoltages)
        time.sleep(1)  # Wait after setting DACs

        # Set the input scaling
        x_scaled = x * genePool.MapGenes(cf.generange[-1], genePool.pool[j, -1])

        # Construct voltage array for fitness function
        V[:2] = x_scaled 
        for control in range(2, 7):
            V[control] = controlVoltages[control-2]*np.ones(len(x[0]))/1E3

        # Save voltage array
        inputTemp[j] = V

        # Measure cf.fitnessavg times the current configuration
        for avgIndex in range(cf.fitnessavg):
            # Get raw voltages
            output = InstrumentImporter.nidaqIO.IO(x_scaled, cf.fs, 
                    inputPorts = [1, 1, 1, 1, 1, 1, 1, 1])

            # Convert voltages to currents (in A)
            for kk in range(2):
                output[kk] = (x_scaled[kk] - output[kk])/cf.resistance
            for kk in range(2,7):
                output[kk] = (controlVoltages[kk-2]/1E3 - output[kk])/cf.resistance
            output[7] = output[7]*cf.amplification/1E9

            # Plot genome
            PlotBuilder.currentGenomeEvolution(mainFig, genePool.pool[j])

            # Train output
            outputAvg[avgIndex] = output

            # Calculate fitness
            fitnessTemp[j, avgIndex]= cf.Fitness(outputAvg[avgIndex],
                                                 V,
                                                 target,
                                                 w,
                                                 cf.P_max)

            # Plot output
            PlotBuilder.currentOutputEvolution(mainFig,
                                               t,
                                               target,
                                               output[7],
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
    inputArray[i, :, :] = inputTemp
    fitnessArray[i, :] = genePool.fitness

    # Update main figure
    PlotBuilder.updateMainFigEvolution(mainFig,
                                       geneArray,
                                       fitnessArray,
                                       outputArray[:, :, 7],
                                       i + 1,
                                       t,
                                       cf.amplification*target,
                                       output,
                                       w)

    # Save generation
    SaveLib.saveExperiment(saveDirectory,
                     geneArray = geneArray,
                     outputArray = outputArray,
                     fitnessArray = fitnessArray,
                     t = t,
                     x = x,
                     amplified_target = cf.amplification*target)

    # Evolve to the next generation
    genePool.NextGen()

PlotBuilder.finalMain(mainFig)

InstrumentImporter.reset(0, 0)

