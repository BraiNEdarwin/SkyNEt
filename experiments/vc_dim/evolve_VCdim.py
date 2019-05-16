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
import pdb

#%% Function definition
def evolve(inputs, binary_labels, 
           path_2_NN = r'/home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/Devices/Marks_Data/April_2019/MSE_n_d10w90_200ep_lr1e-3_b1024_b1b2_0.90.75.pt',
           noise = 0, 
           filepath = r'../../test/evolution_test/VCdim_testing/',
           hush=True):
    
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
    saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)
    
    # Initialize main figure
    if not hush:
        mainFig = PlotBuilder.initMainFigEvolution(cf.genes, cf.generations, cf.genelabels, cf.generange)
    
    # Initialize NN
    net = staNNet(path_2_NN) 
#    pdb.set_trace()
    dtype = torch.FloatTensor#torch.cuda.FloatTensor
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
                g = np.ones_like(target)[:,np.newaxis]*genePool.pool[j,:5][:,np.newaxis].T
                x_dummy = np.concatenate((x_scaled.T,g),axis=1) # First input then genes; dims of input TxD
                inputs = torch.from_numpy(x_dummy).type(dtype)
                inputs = Variable(inputs)
                pdb.set_trace()
                output = net.outputs(inputs)
    
                # Plot genome
                try:
                    PlotBuilder.currentGenomeEvolution(mainFig, genePool.pool[j])
                except:
                    pass
                
                # Train output
                outputAvg[avgIndex] = np.asarray(output) #cf.amplification * np.asarray(output) + noise*(3/100)*(1 + np.abs(np.asarray(output)))*np.random.standard_normal(output.shape) # empty for now, as we have only one output node
                noisy_target = target #+ 0.001*np.random.standard_normal(output.shape)
    
                # Calculate fitness
                fitnessTemp[j, avgIndex]= cf.Fitness(outputAvg[avgIndex],
                                                         noisy_target,
                                                         w)
    
                # Plot output
                try:
                    PlotBuilder.currentOutputEvolution(mainFig,
                                                       t,
                                                       target,
                                                       output,
                                                       j + 1, i + 1,
                                                       fitnessTemp[j, avgIndex])
                except:
                    pass
                
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
        try:
            PlotBuilder.updateMainFigEvolution(mainFig,
                                               geneArray,
                                               fitnessArray,
                                               outputArray,
                                               i + 1,
                                               t,
                                               cf.amplification*target,
                                               output,
                                               w)
        except:
            pass    
        # Save generation
        SaveLib.saveExperiment(saveDirectory,
                               genes = geneArray,
                               output = outputArray,
                               fitness = fitnessArray)
    
        # Evolve to the next generation
        genePool.NextGen()
    
    try:
        PlotBuilder.finalMain(mainFig)
    except:
        pass        
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

    inputs = [[-1.,0.4,-1.,0.4,-0.8, 0.2],[-1.,-1.,0.4,0.4, 0., 0.]]
    binary_labels = [1,0,1,1,0,1]
    best_genome, best_output, max_fitness, accuracy = evolve(inputs,binary_labels,hush=False)