''''
Measurement script to perform an evolution experiment of a selected
gate. This will initially be tested on the Heliox (with nidaq) setup.
'''

# Import packages
import modules.ReservoirFull as Reservoir
import modules.PlotBuilder as PlotBuilder
import modules.GenerateInput as GenerateInput
import modules.Evolution as Evolution
import modules.PostProcess as PostProcess
import modules.SaveLib as SaveLib
from instruments.niDAQ import nidaqIO
from instruments.DAC import IVVIrack
import matplotlib.pyplot as plt

# temporary imports
import numpy as np
import scipy.fftpack



# Read config.txt file
exec(open("config.txt").read())

# initialize genepool
#genePool = Evolution.GenePool(genes, genomes)

# initialize benchmark
# Obtain benchmark input (P and Q are input1, input2)
[t, inp1] = GenerateInput.softwareInput(benchmark, SampleFreq, WavePeriods, WaveFrequency)

WavePeriods2 = (WavePeriods/WaveFrequency)*WaveFrequency2

[t, inp2] = GenerateInput.softwareInput(benchmark, SampleFreq, WavePeriods2, WaveFrequency2)

inp = np.empty((2,len(inp2)))
inp[0,:]=inp1
inp[1,:]=inp2
# format for nidaq
# x = np.empty((2, len(P)))
# x[0,:] = P * 0.1
# x[1,:] = Q * 0.1
# Obtain benchmark target
#[t, target] = GenerateInput.targetOutput(
#    benchmark, SampleFreq, WavePeriods, WaveFrequency)

# np arrays to save genePools, outputs and fitness
# geneArray = np.empty((generations, genes, genomes))
# outputArray = np.empty((generations, len(P) - skipstates, genomes))
# fitnessArray = np.empty((generations, genomes))

# temporary arrays, overwritten each generation
# fitnessTemp = np.empty((genomes, fitnessAvg))
# trained_output = np.empty((len(P) - skipstates, fitnessAvg))
# outputTemp = np.empty((len(P) - skipstates, genomes))
# controlVoltages = np.empty(genes)

# initialize save directory
# saveDirectory = SaveLib.createSaveDirectory(filepath, name)

# initialize main figure
# mainFig = PlotBuilder.initMainFigEvolution(genes, generations, genelabels, generange)


# initialize instruments
#ivvi = IVVIrack.initInstrument()

# feed input to adwin
output = nidaqIO.IO_2D(inp, SampleFreq)
# 2.0/len(output) * np.abs(y[:len(output)//2])
#plot output


plt.plot(t, output)

plt.show()


# y = scipy.fftpack.fft(output)
# x = np.linspace(0.0, 1.0/(2.0*(1/1000)), len(output)/2)
# plt.plot(x, 2.0/len(output) * np.abs(y[:len(output)//2]))
# plt.show()
'''
for i in range(generations):

    for j in range(genomes):

        # set the DAC voltages
        for k in range(genes):
            controlVoltages[k] = Evolution.mapGenes(
                generange[k], genePool.pool[k, j])
        IVVIrack.setControlVoltages(ivvi, controlVoltages)

        for avgIndex in range(fitnessAvg):

            # feed input to adwin
            output = nidaqIO.IO_2D(x, SampleFreq)

            # plot genome
            PlotBuilder.currentGenomeEvolution(mainFig, genePool.pool[:, j])
            
            # Train output
            trained_output[:, avgIndex] = output  # empty for now, as we have only one output node

            # Calculate fitness
            fitnessTemp[j, avgIndex] = PostProcess.fitness(
                trained_output[:, avgIndex], target[skipstates:])

            #plot output
            PlotBuilder.currentOutputEvolution(mainFig, t, target, output, j + 1, i + 1, fitnessTemp[j, avgIndex])

        outputTemp[:, j] = trained_output[:, np.argmin(fitnessTemp[j, :])]

    genePool.fitness = fitnessTemp.min(1)
    print("Generation nr. " + str(i + 1) + " completed")
    print("Highest fitness: " + str(max(genePool.fitness)))

    # save generation data
    geneArray[i, :, :] = genePool.returnPool()
    outputArray[i, :, :] = outputTemp
    fitnessArray[i, :] = fitnessTemp.min(1)

    PlotBuilder.updateMainFigEvolution(mainFig, geneArray, fitnessArray, outputArray, i + 1, t, target, output)
	
	#save generation
    SaveLib.saveMain(saveDirectory, geneArray, outputArray, fitnessArray, t, x, target)
	
    # evolve the next generation
    genePool.nextGen()
'''

np.savetxt('D:/data/BramdW/HH/test_in_1_in_3_8.5Hz_18.5Hz_out_11', (output,t)) 

# PlotBuilder.finalMain(mainFig)
