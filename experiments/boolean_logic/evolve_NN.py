'''
This is a template for evolving the NN based on the boolean_logic experiment. 
The only difference to the measurement scripts are on lines where the device is called.

'''
# SkyNEt imports
import SkyNEt.modules.SaveLib as SaveLib
import SkyNEt.modules.Evolution as Evolution
from SkyNEt.modules.PlotBuilder import PlotBuilder

import config_evolve_NN as config
from SkyNEt.modules.Nets.staNNet import staNNet 
# Other imports
import torch
from torch.autograd import Variable
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

# Initialize figure
pb = PlotBuilder()
pb.add_subplot((0,0), (5, 1), adaptive=True, ylim=(-0.1,1.1), title='cv genes')
pb.add_subplot((0,1), (2 ,t.shape[0]), ylim=(-0.1,1.1), title='current device output')
pb.add_subplot((1,1), cf.genes, ylim=(0,1), title='current genome voltages')
pb.add_subplot((1,0), cf.genes, adaptive=True, title='best fitness')
pb.finalize()

# Initialize NN
main_dir = r'../../test/NN_test/data4nn/Data_for_testing/' 
# NN model coming from /home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/Devices/25_07_2018_CP-full-search-77K/lr2e-4_eps1000_mb512_25072018CP.pt
dtype = torch.FloatTensor
net = staNNet(main_dir+'TEST_NN.pt')

# Initialize genepool
genePool = Evolution.GenePool(cf)

#%% Measurement loop

for i in range(cf.generations):
    start = time.time()
    for j in range(cf.genomes):
        # Set the DAC voltages
        for k in range(cf.genes-1):
            controlVoltages[k] = genePool.MapGenes(
                                    cf.generange[k], genePool.pool[j, k])

        # Set the input scaling
        x_scaled = x * genePool.MapGenes(cf.generange[-1], genePool.pool[j, -1])

        # Measure cf.fitnessavg times the current configuration
        for avgIndex in range(cf.fitnessavg):
            # Feed input to NN
            g = np.ones_like(target)[:,np.newaxis]*genePool.pool[j,:-1][:,np.newaxis].T
            x_dummy = np.concatenate((x_scaled.T,g),axis=1) # First input then genes; dims of input TxD
            inputs = torch.from_numpy(x_dummy).type(dtype)
            inputs = Variable(inputs)
            output = net.outputs(inputs)

            # Plot current genome
            pb.update((1,1), genePool.pool[j])

            # Train output
            outputAvg[avgIndex] = cf.amplification * np.asarray(output)  # empty for now, as we have only one output node

            # Calculate fitness
            fitnessTemp[j, avgIndex]= cf.Fitness(outputAvg[avgIndex],
                                                     target,
                                                     w)

            # Plot current output vs target
            pb.update((0,1), np.stack((target, output)))

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
    best_fitness_index = np.argmax(fitnessArray[:i+1], axis=1)
    
    # Update main figure
    pb.update((0,0), geneArray[np.arange(i+1),best_fitness_index,:].T)
    pb.update((1,0), fitnessArray[np.arange(i+1), best_fitness_index])


    # Save generation
    SaveLib.saveExperiment(saveDirectory,
                           genes = geneArray,
                           output = outputArray,
                           fitness = fitnessArray)

    # Evolve to the next generation
    genePool.NextGen()