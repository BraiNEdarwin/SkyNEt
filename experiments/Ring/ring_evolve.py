'''
This is a template for evolving the NN based on the boolean_logic experiment. 
The only difference to the measurement scripts are on lines where the device is called.

'''
# General imports
import signal
import sys
import time
import numpy as np
import pdb
import logging
import torch
# SkyNEt imports
import SkyNEt.modules.SaveLib as SaveLib
import SkyNEt.modules.Evolution_New as Evolution
import SkyNEt.modules.PlotBuilder as PlotBuilder
import config_ring as config
from SkyNEt.modules.Classifiers import perceptron
from SkyNEt.modules.Nets.staNNet import staNNet 
import os



#%%
def MapGenes(generange, gene):
        '''Convert the gene [0,1] to the appropriate value set by generange [a,b]'''
        return generange[0] + gene * (generange[1] - generange[0])

def evolve(inputs, binary_labels,
           filepath = r'../../test/evolution_test/Ring_testing/', hush=True, save=False, showbest =True ):
    #signal.signal(signal.SIGINT, reset)
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
    #controlVoltages = np.zeros(cf.genes)
    
    if save:
        # Initialize save directory
        saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)
    
    # Initialize main figure
    if not hush or showbest:
        mainFig = PlotBuilder.initMainFigEvolution(cf.genes, cf.generations, cf.genelabels, cf.generange)
    else:
        print('WARNING: Plot is hushed; change hush flag to False for plotting...')
   
    # Initialize NN
    main_dir = r'../../test/NN_test/data4nn/Data_for_testing/'
    dtype = torch.FloatTensor
    net = staNNet(main_dir+'NN_New.pt')

    
    # Initialize genepool
    genePool = Evolution.GenePool(cf)
    #Note: genePool consists of 8 genes (you will change the shift and input scaling as well. )
    #%% Measurement loop
    for i in range(cf.generations):
        start = time.time()
        for j in range(cf.genomes):
            # Set the DAC voltages
#            for k in range(cf.genes-3):
#                controlVoltages[k] = genePool.MapGenes(
#                                        cf.generange[k], genePool.pool[j, k])

            #Scale the input (normalize in put to [0,1] and then map to the input range)
            x_normalized = (x-x.min())/(x.max()-x.min())
            #map input to input range 
            x_mapped = MapGenes(cf.inputrange, x_normalized)
            #note input is already in V 
            x_scaled = x_mapped * genePool.pool[j, -3]
            
#            # Set the input offset
            s1 = genePool.pool[j, -2]
            s2 = genePool.pool[j, -1]
            shift = np.array([s1,s2])[:,np.newaxis]
#            pdb.set_trace()
            x_scaled += shift
            
            
            # Measure cf.fitnessavg times the current configuration
            for avgIndex in range(cf.fitnessavg):
                # Feed input to NN
                #Note: add 0:5 because otherwise you only want to feed the control voltages concatenated with the inputs
                g = np.ones_like(target)[:,np.newaxis]*genePool.pool[j][0:5,np.newaxis].T
                if (cf.input_electrodes == [0,1] or cf.input_electrodes == [2,3] or cf.input_electrodes == [4,5]):
                    x_dummy = np.insert(g, cf.input_electrodes[0], x_scaled.T[:,0],axis=1)
                    x_dummy = np.insert(x_dummy, cf.input_electrodes[1], x_scaled.T[:,1],axis=1)
                else:
                    print("Combination input electrodes is not valid")
                    time.sleep(1)
                    os._exit(1)
                        
               
                inputs = torch.from_numpy(x_dummy).type(dtype)
                #inputs = Variable(inputs)
            
                output = net.outputs(inputs)
    
                # Plot genome
                try:
                    if not hush: PlotBuilder.currentGenomeEvolution(mainFig, genePool.pool[j])
                except:
                    if j == 0:
                        logging.exception('PlotBuilder.currentGenomeEvolution FAILED!')
                        print('Gene pool shape is ',genePool.pool.shape)
    
                # Train output
                outputAvg[avgIndex] = cf.amplification * output
                # Calculate fitness
                fitnessTemp[j, avgIndex]= cf.Fitness(outputAvg[avgIndex],
                                                         target,
                                                         w)
                # Plot output
                try:
                    if not hush:PlotBuilder.currentOutputEvolution(mainFig,
                                                       t,
                                                       target,
                                                       output,
                                                       j + 1, i + 1,
                                                       fitnessTemp[j, avgIndex])
                except:
                    if j == 0: logging.exception('PlotBuilder.currentOutputEvolution FAILED!')
                
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
            if not hush: logging.exception('PlotBuilder.updateMainFigEvolution FAILED!')
        
        if save:
            # Save generation
            SaveLib.saveExperiment(saveDirectory,
                                   genes = geneArray,
                                   output = outputArray,
                                   fitness = fitnessArray,
                                   target = target[w][:,np.newaxis])
    
        # Evolve to the next generation
        genePool.NextGen()
    
    
    try:
        PlotBuilder.finalMain(mainFig)
    except:
        if not hush: logging.exception('WARNING: PlotBuilder.finalMain FAILED!')
    
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
      
#%% MAIN
if __name__=='__main__':
    
    from matplotlib import pyplot as plt
    with np.load('Class_data_0.40.npz') as data:
        inputs = data['inp_wvfrm'].T
        labels = data['target']

    cf = config.experiment_config(inputs, labels)
    target_wave = cf.TargetGen
    t, inp_wave, weights = cf.InputGen
    plt.figure()
    plt.plot(t,inp_wave.T)
    plt.plot(t,target_wave,'k')
    plt.show()
#    print(sys.path)
    _,_,_,_ = evolve(inputs,labels)
    #_,_,_= evolve(inputs,labels)