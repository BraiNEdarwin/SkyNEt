#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:59:05 2019

@author: annefleur
"""

"""
Script to evolve the NN using the Evolution_Final.py module. 
It contains 2 functions: a noise generator and a function to compare the opposite pool with the current pool
Note: the search automatically stops if a linearly separable solution is found. 
If you want to continue the search for higher quality solutions, remove the 'break'
More information about each parameter can be found below
"""

# SkyNEt imports
import SkyNEt.modules.SaveLib as SaveLib
import Evolution_Final as Evolution
import SkyNEt.modules.PlotBuilder as PlotBuilder
import config_evolve_VCdim5_Final as config
from SkyNEt.modules.Nets.staNNet import staNNet 
from SkyNEt.modules.Classifiers import perceptron
import torch
from torch.autograd import Variable
import time
import numpy as np
import pdb
import os 
from SkyNEt.instruments import InstrumentImporter


##Returns the output array with noise 
def noise_gen(output,sigma=0.01, sigma_0=0.1):
    variance = output**2 * sigma**2 + sigma_0**2
    standard_dev = np.sqrt(variance)
    noise = np.random.normal(0, standard_dev)
    y = output + noise
    return y, standard_dev

##Compares the current pool with the opposite pool. If fit(genome oppositepool)>
## fit(genome currentpool) the genome in the currentpool is replaced. 
def opposite( x, genePool, cf, net, target, dtype,w):
    genePool.Opposite()
    list_pool = [genePool.pool, genePool.opposite_pool]
    outputAvg = np.zeros((cf.fitnessavg, len(x[0])))
    temp_list = [np.zeros((cf.genomes, cf.fitnessavg)), np.zeros((cf.genomes, cf.fitnessavg))]
    for i in range(len(list_pool)):
        for j in range(cf.genomes):
            x_scaled = x * genePool.config_obj.input_scaling
            for avgIndex in range(cf.fitnessavg):
                g = np.ones_like(target)[:,np.newaxis]*list_pool[i][j][:,np.newaxis].T
                if (cf.input_electrodes == [1,2] or cf.input_electrodes == [3,4] or cf.input_electrodes == [5,6]):
                    x_dummy = np.insert(g, cf.input_electrodes[0], x_scaled.T[:,0],axis=1)
                    x_dummy = np.insert(x_dummy, cf.input_electrodes[1], x_scaled.T[:,1],axis=1)
                else:
                    print("Combination input electrodes is not valid")
                    time.sleep(1)
                    os._exit(1)
                if cf.use_nn:
                    inputs = torch.from_numpy(x_dummy).type(dtype)
                    inputs = Variable(inputs)
                    output = net.outputs(inputs) * cf.amplification_nn 
                    # Add Noise
                    output, standard_dev = noise_gen(output)
                else:
                    inputs = x_dummy.T
                    output = InstrumentImporter.nidaqIO.IO_cDAQ(inputs, cf.fs) *cf.amplification_chip
                outputAvg[avgIndex] =  np.asarray(output) 
                temp_list[i][j,avgIndex] = cf.Fitness(outputAvg[avgIndex],target, w)
    indices = temp_list[1] > temp_list[0]
    genePool.setNewPool(indices)     


def evolve( dataset, threshold, inputs, binary_labels):
    # Initialize config object
    cf = config.experiment_config(inputs, binary_labels)
    # Initialize input and target
    t = cf.InputGen[0]  # Time array
    x = cf.InputGen[1]  # Array with P and Q signal
    w = cf.InputGen[2]  # Weight array
    target = cf.TargetGen  # Target signal
    
    #arrays to save genePools, outputs and fitness
    geneArray = np.zeros((cf.generations, cf.genomes, cf.genes))
    outputArray = np.zeros((cf.generations, cf.genomes, len(x[0])))
    fitnessArray = np.zeros((cf.generations, cf.genomes))
    # Temporary arrays, overwritten each generation
    fitnessTemp = np.zeros((cf.genomes, cf.fitnessavg))
    outputAvg = np.zeros((cf.fitnessavg, len(x[0])))
    outputTemp = np.zeros((cf.genomes, len(x[0])))

    # Initialize NN
    if cf.use_nn:
        main_dir = r'../../test/NN_test/data4nn/Data_for_testing/'
        dtype = torch.FloatTensor
        net = staNNet(main_dir+'NN_New2.pt')  
    else:
        net = 0
        dtype = 0
        main_dir = 0
    
    # Initialize genepool  
    genePool = Evolution.GenePool(cf)  

    #%% Measurement loop

    for i in range(cf.generations):
        start = time.time()
        #if the solution is not found when half of the generations have elapsed, evaluate opposite pool
        if (i+1)%40 == 0: 
            opposite( x, genePool, cf, net, target, dtype,w)

        for j in range(cf.genomes):
            x_scaled = x * genePool.config_obj.input_scaling
            # Measure cf.fitnessavg times the current configuration
            for avgIndex in range(cf.fitnessavg):
                # Feed input to NN
                g = np.ones_like(target)[:,np.newaxis]*genePool.pool[j][:,np.newaxis].T
                if (cf.input_electrodes == [1,2] or cf.input_electrodes == [3,4] or cf.input_electrodes == [5,6]):
                    x_dummy = np.insert(g, cf.input_electrodes[0], x_scaled.T[:,0],axis=1)
                    x_dummy = np.insert(x_dummy, cf.input_electrodes[1], x_scaled.T[:,1],axis=1)
                else:
                    print("Combination input electrodes is not valid")
                    time.sleep(1)
                    os._exit(1)               
                if cf.use_nn:
                    inputs = torch.from_numpy(x_dummy).type(dtype)
                    inputs = Variable(inputs)
                    output = net.outputs(inputs) * cf.amplification_nn 
                    # Add Noise
                    output, standard_dev = noise_gen(output)
                else:
                    inputs = x_dummy.T
                    output = InstrumentImporter.nidaqIO.IO_cDAQ(inputs, cf.fs) *cf.amplification_chip
                outputAvg[avgIndex] =  np.asarray(output) 
                # Calculate fitness
                fitnessTemp[j, avgIndex]= cf.Fitness(outputAvg[avgIndex],target, w)
    
            outputTemp[j] = outputAvg[np.argmin(fitnessTemp[j])]
        #Save fitness
        genePool.fitness = fitnessTemp.min(1) 
        #Stop the time and print status
        end = time.time()
        print("Generation nr. " + str(i + 1) + " completed; took "+str(end-start)+" sec.")
        print("Highest fitness: " + str(max(genePool.fitness)))

        # generation data
        geneArray[i, :, :] = genePool.pool
        outputArray[i, :, :] = outputTemp
        fitnessArray[i, :] = genePool.fitness
        ind = np.unravel_index(np.argmax(fitnessArray, axis=None), fitnessArray.shape)
        best_genome = geneArray[ind]
        best_output = outputArray[ind]
        y = best_output[w][:,np.newaxis]
        trgt = target[w][:,np.newaxis] 
        #If one of the genomes is linearly separable, break. See methods in config_evolve for 
        #A more detailed description of this threshold. 
        if max(genePool.fitness) > 0.1:
            end = i 
            break 
        #Evolve to next generation 
        genePool.NextGen(i)
    #Get best results
    max_fitness = np.max(fitnessArray)
    a = fitnessArray
    ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
    assert fitnessArray[ind]==max_fitness,'Indices do not give value'
    best_genome = geneArray[ind]
    best_output = outputArray[ind]
    y = best_output[w][:,np.newaxis]
    trgt = target[w][:,np.newaxis]
    accuracy, _, _ = perceptron(y,trgt)
    print('Max. Fitness: ', max_fitness)
    print('Best genome: ', best_genome)
    print('Accuracy of best genome: ', accuracy)
    #Return relevant data 
    return best_genome, best_output, max_fitness, accuracy, cf.TargetGen, end,w 
