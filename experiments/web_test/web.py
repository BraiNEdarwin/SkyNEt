
"""
TODO:
- Plotbuilder:
    axis labels, tightlayout, red color, minor and major grid, xticks
"""


execute = False

# SkyNEt imports
import SkyNEt.modules.SaveLib as SaveLib
import SkyNEt.modules.Evolution as Evolution
from SkyNEt.modules.PlotBuilder import PlotBuilder
import config_web as config
from SkyNEt.instruments import InstrumentImporter
from SkyNEt.instruments.Keithley2400 import Keithley2400

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
outputArray = np.zeros((cf.generations, cf.genomes, len(x[0])))
fitnessArray = np.zeros((cf.generations, cf.genomes))

# Temporary arrays, overwritten each generation
fitnessTemp = np.zeros((cf.genomes, cf.fitnessavg))
outputAvg = np.zeros((cf.fitnessavg, len(x[0])))
outputTemp = np.zeros((cf.genomes, len(x[0])))
controlVoltages = np.zeros(cf.genes-1)

if execute:
    # Initialize save directory
    saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# Initialize main figure
pb = PlotBuilder()
pb.add_subplot('genes',     (0,0), (5, 1),   adaptive=True,  ylim=(-0.1,1.1), title='History of best genes',    xlabel='generations', rowspan=2, legend=cf.genelabels)
pb.add_subplot('fitness',   (2,0), cf.genes, adaptive=True,                   title='History of best fitness',  xlabel='generations',  rowspan=2)
pb.add_subplot('cur_output',(0,1), t.shape[0], adaptive=True,               title='Fittest device output of last generation', rowspan=2)
pb.add_subplot('cur_genome',(2,1), cf.genes,                 ylim=(0,1),      title='Current genome voltages')
pb.add_subplot('output',    (3,1), t.shape[0], adaptive=True,               title='Device output')
pb.finalize()

if execute:
    # Initialize instruments
    #ivvi = InstrumentImporter.IVVIrack.initInstrument()
    keithley = Keithley2400.Keithley_2400('keithley', 'GPIB0::11')
    keithley.compliancei.set(1E-6)
    keithley.compliancev.set(4)
    keithley.output.set(1)
    # arduino switch network
    # Initialize serial object
    ser = InstrumentImporter.switch_utils.init_serial(cf.switch_comport)
    # Switch to device
    InstrumentImporter.switch_utils.connect_single_device(ser, cf.switch_device)
    print('INFO: Connected device %i' % cf.switch_device)

    ivvi = InstrumentImporter.IVVIrack.initInstrument()

# Initialize genepool
genePool = Evolution.GenePool(cf)

#%% Measurement loop
for i in range(cf.generations):
    for j in range(cf.genomes):
        # Set the DAC voltages
        for k in range(cf.genes-1):
            controlVoltages[k] = genePool.MapGenes(
                                    cf.generange[k], genePool.pool[j, k])/1000

        # Set the input scaling
        x_scaled = x * genePool.MapGenes(cf.generange[-1], genePool.pool[j, -1])
        

        # Measure cf.fitnessavg times the current configuration
        for avgIndex in range(cf.fitnessavg):
            output = np.zeros(4)
            if execute:
                for iii in range(4):
                    # [I0, CV0, CV1, CV2, CV3, I1, 0]
                    input_voltages_switch = np.concatenate((np.array([x_scaled[0,iii]]),controlVoltages,np.array([x_scaled[1,iii], 0.])))
                    InstrumentImporter.IVVIrack.setControlVoltages(ivvi, input_voltages_switch)
                    output[iii] = -keithley.curr()*1e9 # in nA
            else:
                output = (np.random.rand(4)-0.5)*2*30

            # Train output
            outputAvg[avgIndex] = cf.amplification * np.asarray(output)  # empty for now, as we have only one output node

            # Calculate fitness
            fitnessTemp[j, avgIndex]= cf.Fitness(outputAvg[avgIndex],
                                                     target,
                                                     w)
            print('generation: %i, genome: %i, fitness %0.3f' % (i,j,fitnessTemp[j,avgIndex]))

        outputTemp[j] = outputAvg[np.argmin(fitnessTemp[j])]
        # Plot current genome
        pb.update('cur_genome', genePool.pool[j])
        # Plot current device output
        pb.update('output', outputTemp[j])# np.stack((target, outputTemp[j])))

    genePool.fitness = fitnessTemp.min(1)  # Save fitness

    # Status print
    print("Generation nr. " + str(i + 1) + " completed")
    print("Highest fitness: " + str(max(genePool.fitness)))

    # Save generation data
    geneArray[i, :, :] = genePool.pool
    outputArray[i, :, :] = outputTemp
    fitnessArray[i, :] = genePool.fitness

    best_fitness_index = np.argmax(fitnessArray[:i+1], axis=1)
    # Update main figure
    # Plot best output of last generation
    pb.update('cur_output', outputTemp[best_fitness_index[i]])#np.stack((target, outputTemp[best_fitness_index[0]])))
    # Plot history of genes
    pb.update('genes', geneArray[np.arange(i+1),best_fitness_index,:].T)
    # Plot best fitness of each generation
    pb.update('fitness', fitnessArray[np.arange(i+1), best_fitness_index])

    if execute:
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

if execute:
    InstrumentImporter.switch_utils.close(ser)
    keithley.output.set(0)
    keithley.close()

    InstrumentImporter.reset(0, 0)
