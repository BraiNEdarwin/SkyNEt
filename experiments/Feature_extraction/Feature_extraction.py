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
import SkyNEt.modules.PlotBuilder as PlotBuilder
import SkyNEt.experiments.Feature_extraction.config_Feature_extraction as config
from SkyNEt.instruments import InstrumentImporter

# Other imports
import time
import numpy as np

# Initialize config object
cf = config.experiment_config()

# Initialize input and target
inp = cf.InputGen()
meas = np.zeros(cf.measurelength)
# np arrays to save genePools, outputs and fitness
geneArray = np.zeros((cf.generations, cf.genomes, cf.genes))
outputArray = np.zeros((cf.generations, cf.genomes, len(inp), cf.measurelength))
fitnessArray = np.zeros((cf.generations, cf.genomes))

# Temporary arrays, overwritten each generation
fitnessTemp = np.zeros((cf.genomes, cf.fitnessavg))
outputAvg = np.zeros((cf.fitnessavg, len(inp),cf.measurelength))
outputTemp = np.zeros((cf.genomes, len(inp),cf.measurelength))
controlVoltages = np.zeros(7)
controls = np.zeros(3)
output = np.zeros((len(inp), cf.measurelength))
# Initialize save directory
saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# Initialize main figure
mainFig = PlotBuilder.initMainFigEvolution(cf.genes, cf.generations, cf.genelabels, cf.generange)

# Initialize instruments
ivvi = InstrumentImporter.IVVIrack.initInstrument()

# Initialize genepool
genePool = Evolution.GenePool(cf)

#%% Measurement loop

for i in range(cf.generations):
    for j in range(cf.genomes):
        # Set the DAC voltages
        contorls = np.zeros(cf.genes)
        for k in range(cf.genes):
            controls[k] = genePool.MapGenes(
                                    cf.generange[k], genePool.pool[j, k])
        controlVoltages[4:] = controls[:3]
        InstrumentImporter.IVVIrack.setControlVoltages(ivvi, controlVoltages)
        print("current CV1,CV2,CV3")
        print(controlVoltages[4:])

        # Set the input scaling
        #x_scaled = x * genePool.MapGenes(cf.generange[-1], genePool.pool[j, -1])

        # Measure cf.fitnessavg times the current configuration
        for avgIndex in range(cf.fitnessavg):
            # set inputs to dacs
            for n in range(len(inp)):
              inputVoltages = [(inp[n,0])*1000-500, (inp[n,1])*1000-500, (inp[n,2])*1000-500, (inp[n,3])*1000-500]
              print("current input")
              print(inputVoltages)
              InstrumentImporter.IVVIrack.setControlVoltages(ivvi, inputVoltages)
              time.sleep(0.2)
              for m in range(cf.ntest):
            # Feed input to measurement device
                if(cf.device == 'nidaq'):
                    measureddata = np.asarray(InstrumentImporter.nidaqIO.IO(meas, cf.fs))*cf.amplification
                elif(cf.device == 'adwin'):
                    adw = InstrumentImporter.adwinIO.initInstrument()
                    measureddata = InstrumentImporter.adwinIO.IO(adw, meas, cf.fs)
                else:
                    print('Specify measurement device as either adwin or nidaq')
                output[n,0:cf.measurelength] = measureddata

            # Plot genome
            # PlotBuilder.currentGenomeEvolution(mainFig, genePool.pool[j])

            # Train output
            outputAvg[avgIndex] = cf.amplification * np.asarray(output)  # empty for now, as we have only one output node

            # Calculate fitness
            fitnessTemp[j, avgIndex]= cf.Fitness(output,
                                                     cf.TargetGen,)
            # Plot output
            # PlotBuilder.currentOutputClassification(mainFig,
            #                                    output,
            #                                    inp,
            #                                    j + 1, i + 1,
            #                                    fitnessTemp[j, avgIndex],
            #                                    cf.TargetGen)
            print(fitnessTemp[j, avgIndex])

        outputTemp[j] = outputAvg[np.argmin(fitnessTemp[j])]

    genePool.fitness = fitnessTemp.min(1)  # Save fitness

    # Status print
    print("Generation nr. " + str(i + 1) + " completed")
    print("Highest fitness: " + str(max(genePool.fitness)))

    # Save generation data
    geneArray[i, :, :] = genePool.pool
    outputArray[i, :, :] = outputTemp
    fitnessArray[i, :] = genePool.fitness

    # Update main figure
    PlotBuilder.updateMainFigClassification(mainFig,
                                       geneArray,
                                       fitnessArray,
                                       outputArray,
                                       i + 1,
                                       inp,
                                       cf.TargetGen)

    # Save generation
    SaveLib.saveExperiment(saveDirectory,
                     geneArray = geneArray,
                     outputArray = outputArray,
                     fitnessArray = fitnessArray,
                     input = inp,
                     target = cf.TargetGen)

    # Evolve to the next generation
    genePool.NextGen()

PlotBuilder.finalMain(mainFig)

InstrumentImporter.reset(0, 0)

