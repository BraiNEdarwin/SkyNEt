'''
This is a template for evolving the NN based on the boolean_logic experiment. 
The only difference to the measurement scripts are on lines where the device is called.

'''
# SkyNEt imports
import SkyNEt.modules.SaveLib as SaveLib
import SkyNEt.modules.Evolution as Evolution
import SkyNEt.modules.PlotBuilder as PlotBuilder
import config_evolve_VCdim as config
from SkyNEt.instruments.niDAQ import nidaqIO
from SkyNEt.instruments.DAC import IVVIrack
# Other imports
import signal
import sys
import time
import numpy as np

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
        
signal.signal(signal.SIGINT, reset)

#%% Initialization
inputs = [[-1,1,-1,1],[-1,-1,1,1]]
binary_labels = [0,10,10,0]

# Initialize config object
cf = config.experiment_config(inputs,binary_labels)

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
mainFig = PlotBuilder.initMainFigEvolution(cf.genes, cf.generations, cf.genelabels, cf.generange)

# Initialize instruments
ivvi = IVVIrack.initInstrument()

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
        IVVIrack.setControlVoltages(ivvi, controlVoltages)
        time.sleep(1)  # Wait after setting DACs
        
        # Set the input scaling
        x_scaled = 2 * (x - 0.5) * genePool.config_obj.input_scaling

        # Measure cf.fitnessavg times the current configuration
        for avgIndex in range(cf.fitnessavg):
            # Feed input to niDAQ
            output = nidaqIO.IO_2D(x_scaled, cf.fs)

            # Plot genome
            PlotBuilder.currentGenomeEvolution(mainFig, genePool.pool[j])

            # Train output
            outputAvg[avgIndex] = cf.amplification / cf.postgain * np.asarray(output) + 0.05*(0.5+np.asarray(output))*np.random.standard_normal(np.array(output).shape) # empty for now, as we have only one output node

            # Calculate fitness
            fitnessTemp[j, avgIndex]= cf.Fitness(outputAvg[avgIndex],
                                                     target,
                                                     w)

            # Plot output
            PlotBuilder.currentOutputEvolution(mainFig,
                                               t,
                                               target,
                                               cf.amplification / cf.postgain * np.array(output),
                                               j + 1, i + 1,
                                               fitnessTemp[j, avgIndex])

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
                                       target,
                                       cf.amplification / cf.postgain * np.array(output),
                                       w)

    # Save generation
    SaveLib.saveExperiment(cf.configSrc, saveDirectory,
                           genes = geneArray,
                           output = outputArray,
                           fitness = fitnessArray)

    # Evolve to the next generation
    genePool.NextGen()

PlotBuilder.finalMain(mainFig)

reset(0, 0)