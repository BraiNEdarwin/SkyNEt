'''
This is a template for evolving the NN based on the boolean_logic experiment. 
The only difference to the measurement scripts are on lines where the device is called.

'''
# SkyNEt imports
import SkyNEt.modules.SaveLib as SaveLib
import SkyNEt.modules.Evolution as Evolution
import SkyNEt.modules.PlotBuilder as PlotBuilder
import config_evolve_VCdim as config
from SkyNEt.modules.Nets.staNNet import staNNet 
from SkyNEt.modules.Classifiers import perceptron
# Other imports
import torch
from torch.autograd import Variable
import time
import numpy as np

#%% Function definition
def evolve(inputs, binary_labels, filepath = r'../../test/evolution_test/VCdim_testing/'):
    # Initialize config object
    cf = config.experiment_config(inputs, binary_labels, filepath=filepath)
    
    # Initialize input and target
    t = cf.InputGen[0]  # Time array
    x = cf.InputGen[1]  # Array with P and Q signal
    w = cf.InputGen[2]  # Weight array
    target = cf.TargetGen  # Target signal
    
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
    #saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)
    
    # Initialize main figure
    mainFig = PlotBuilder.initMainFigEvolution(cf.genes, cf.generations, cf.genelabels, cf.generange)
    
    # Initialize NN
    main_dir = r'/home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/2018_08_07_164652_CP_FullSwipe/'
    dtype = torch.cuda.FloatTensor
    net = staNNet(main_dir+'lr2e-4_eps400_mb512_20180807CP.pt')
    
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
            x_scaled = x * genePool.config_obj.input_scaling
    
            # Measure cf.fitnessavg times the current configuration
            for avgIndex in range(cf.fitnessavg):
                # Feed input to NN
                g = np.ones_like(target)[:,np.newaxis]*genePool.pool[j][:,np.newaxis].T
                x_dummy = np.concatenate((x_scaled.T,g),axis=1) # First input then genes; dims of input TxD
                inputs = torch.from_numpy(x_dummy).type(dtype)
                inputs = Variable(inputs)
                output = net.outputs(inputs)
    
    #            # Plot genome
    #            PlotBuilder.currentGenomeEvolution(mainFig, genePool.pool[j])
    
                # Train output
                outputAvg[avgIndex] = cf.amplification * np.asarray(output) + 0.05*(0.5+np.abs(np.asarray(output)))*np.random.standard_normal(output.shape) # empty for now, as we have only one output node
                noisy_target = target + 0.01*np.random.standard_normal(output.shape)
    
                # Calculate fitness
                fitnessTemp[j, avgIndex]= cf.Fitness(outputAvg[avgIndex],
                                                         noisy_target,
                                                         w)
    
    #            # Plot output
    #            PlotBuilder.currentOutputEvolution(mainFig,
    #                                               t,
    #                                               target,
    #                                               output,
    #                                               j + 1, i + 1,
    #                                               fitnessTemp[j, avgIndex])
    
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
    #    SaveLib.saveExperiment(saveDirectory,
    #                           genes = geneArray,
    #                           output = outputArray,
    #                           fitness = fitnessArray)
    
        # Evolve to the next generation
        genePool.NextGen()
    
    PlotBuilder.finalMain(mainFig)
    
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
    return best_genome, best_output, max_fitness, accuracy

#%% Initialization
if __name__=='__main__':

    inputs = [[-1,1,-1,1],[-1,-1,1,1]]
    binary_labels = [1,0,0,1]
    _,_,_,_ = evolve(inputs,binary_labels)