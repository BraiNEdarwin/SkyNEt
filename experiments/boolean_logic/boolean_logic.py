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
import SkyNEt.modules.Evolution as Evolution
from SkyNEt.modules.PlotBuilder import PlotBuilder
import config_boolean_logic as config
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
pb.add_subplot('genes',     (0,0), (5, 1),   adaptive=True,  ylim=(-0.1,1.1), title='History of best genes',    xlabel='generations', rowspan=2)
pb.add_subplot('fitness',   (2,0), cf.genes, adaptive=True,                   title='History of best fitness',  xlabel='generations',  rowspan=2)
pb.add_subplot('cur_output',(0,1), (2 ,t.shape[0]),          ylim=(-0.1,1.1), title='Fittest device output of last generation', legend=['target', 'device'], rowspan=2)
pb.add_subplot('cur_genome',(2,1), cf.genes,                 ylim=(0,1),      title='Current genome voltages')
pb.add_subplot('output',    (3,1), (2 ,t.shape[0]),          ylim=(-0.1,1.1), title='Device output', legend=['target', 'device'])
pb.finalize()

# Initialize instruments
ivvi = InstrumentImporter.IVVIrack.initInstrument()

# Initialize genepool
genePool = Evolution.GenePool(cf)

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

        # Measure cf.fitnessavg times the current configuration
        for avgIndex in range(cf.fitnessavg):
            # Feed input to measurement device
            if(cf.device == 'nidaq'):
                output = InstrumentImporter.nidaqIO.IO_2D(x_scaled, cf.fs)
            elif(cf.device == 'adwin'):
                adw = InstrumentImporter.adwinIO.initInstrument()
                output = InstrumentImporter.adwinIO.IO_2D(adw, x_scaled, cf.fs)
            else:
                print('Specify measurement device as either adwin or nidaq')

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

    # Status print
    print("Generation nr. " + str(i + 1) + " completed")
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
                     geneArray = geneArray,
                     outputArray = outputArray,
                     fitnessArray = fitnessArray,
                     t = t,
                     x = x,
                     amplified_target = cf.amplification*target)

    # Evolve to the next generation
    genePool.NextGen()

InstrumentImporter.reset(0, 0)

