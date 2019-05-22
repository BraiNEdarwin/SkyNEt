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
single_line_prop = {'color':'red', 'marker':'x'}
pb = PlotBuilder()
pb.add_subplot('genes',     (0,0), (5, 1), adaptive='x', ylim=(-0.1,1.1), title='History of best genes',    xlabel='generations', rowspan=2, legend=cf.genelabels)
pb.add_subplot('fitness',   (2,0), 1,      adaptive='both', title='History of best fitness',  xlabel='generations',  rowspan=2, lineprop=single_line_prop)
pb.add_subplot('cur_output',(0,1), (2 ,t.shape[0]),       ylim=(-0.1,1.1), title='Fittest device output of last generation', legend=['target', 'device'], rowspan=2)
pb.add_subplot('cur_genome',(2,1), cf.genes,              ylim=(0,1),      title='Current genome voltages', lineprop=single_line_prop)
pb.add_subplot('output',    (3,1), (2 ,t.shape[0]),       ylim=(-0.1,1.1), title='Device output', legend=['target', 'device'])
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

            # Train output
            outputAvg[avgIndex] = cf.amplification * np.asarray(output)  # empty for now, as we have only one output node

            # Calculate fitness
            fitnessTemp[j, avgIndex]= cf.Fitness(outputAvg[avgIndex],
                                                     target,
                                                     w)


        outputTemp[j] = outputAvg[np.argmin(fitnessTemp[j])]
        
        # Plot current genome
        pb.update('cur_genome', genePool.pool[j])
        # Plot current device output
        pb.update('output', np.stack((target, outputTemp[j])))

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
    
    # Plot best output of last generation
    pb.update('cur_output', np.stack((target, outputTemp[best_fitness_index[0]])))
    # Plot history of genes
    pb.update('genes', geneArray[np.arange(i+1),best_fitness_index,:].T)
    # Plot best fitness of each generation
    pb.update('fitness', fitnessArray[np.arange(i+1), best_fitness_index])


    # Save generation
    SaveLib.saveExperiment(saveDirectory,
                           genes = geneArray,
                           output = outputArray,
                           fitness = fitnessArray)

    # Evolve to the next generation
    genePool.NextGen()