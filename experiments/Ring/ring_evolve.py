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
# SkyNEt imports
import SkyNEt.modules.SaveLib as SaveLib
import SkyNEt.modules.Evolution as Evolution
import SkyNEt.modules.PlotBuilder as PlotBuilder
import config_ring as config
try:
    from SkyNEt.instruments import InstrumentImporter
    from SkyNEt.instruments.DAC import IVVIrack
    from SkyNEt.instruments.ADwin import adwinIO
    importerror = []
except ImportError as error:
    print('############################################')
    print('WARNING:',error,'!!')
    print('WARNING: Random output will be generated!!')
    importerror = error
    print('############################################')
from SkyNEt.modules.Classifiers import perceptron


#%%
def evolve(inputs, binary_labels,
           filepath = r'../../test/evolution_test/Ring_testing/', hush=True):
    signal.signal(signal.SIGINT, reset)
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
    else:
        print('WARNING: Plot is hushed; change hush flag to True for plotting...')
    # Initialize instruments
    if not importerror:
        ivvi = InstrumentImporter.IVVIrack.initInstrument()
        adw = InstrumentImporter.adwinIO.initInstrument()
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
            if not importerror:
                IVVIrack.setControlVoltages(ivvi, controlVoltages)
                time.sleep(1)  # Wait after setting DACs

            # Set the input scaling
#            x_scaled = x * genePool.config_obj.input_scaling
            x_scaled = x * genePool.MapGenes(cf.generange[-3], genePool.pool[j, -3]) 
            # Set the input offset
            s1 = genePool.MapGenes(cf.generange[-2], genePool.pool[j, -2])
            s2 = genePool.MapGenes(cf.generange[-1], genePool.pool[j, -1])
            shift = np.array([s1,s2])[:,np.newaxis]
#            pdb.set_trace()
            x_scaled += shift
            # Measure cf.fitnessavg times the current configuration
            for avgIndex in range(cf.fitnessavg):
                # Feed input to niDAQ
                if not importerror:
                    output = adwinIO.IO(adw, x_scaled, cf.fs)
                    output = np.array(output[0,:])
                else:
                    output = 0.1*np.random.standard_normal(len(x[0]))
                    if j == 0:
                        print('WARNING: Debug mode active; output is white noise!!')
    
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

#%% Help functions
def reset(signum, frame):
        '''
        This functions performs the following reset tasks:
        - Set IVVI rack DACs to zero
        - Apply zero signal to the NI daq
        - Apply zero signal to the ADwin
        '''
        try:
            global ivvi
            ivvi.set_dacs_zero()
            print('ivvi DACs set to zero')
            del ivvi  # Test if this works!
        except:
            print('ivvi was not initialized, so also not reset')
			
        try:
            nidaqIO.reset_device()
            print('nidaq has been reset')
        except:
            print('nidaq not connected to PC, so also not reset')

        try:
            global adw
            reset_signal = np.zeros((2, 40003))
            adwinIO.IO_2D(adw, reset_signal, 1000)
        except:
            print('adwin was not initialized, so also not reset')

        # Finally stop the script execution
        sys.exit()         
#%% MAIN
if __name__=='__main__':
    
    from matplotlib import pyplot as plt
    with np.load('Class_data.npz') as data:
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
    _,_,_,_ = evolve(inputs,labels,hush=False)