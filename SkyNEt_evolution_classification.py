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
import time

# temporary imports
import numpy as np

# print (list(iter.product(a, repeat=6)))

# Read config.txt file
exec(open("config.txt").read())

# initialize genepool
genePool = Evolution.GenePool(genes, genomes)

# initialize benchmark
# Obtain benchmark input (P and Q are input1, input2)
# win is the convolution window, 16 possiblities
import itertools

a = [0, 1]
b = [-1,0,1]
c = 24
win = np.array(list(itertools.product(*[a,a,a,b])))

# win = np.array([[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],[1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1],[-1,0,0,0],[-1,1,0,0]])
wtest = 10
fs = 1000
vin=800
y = np.zeros([100])
winout = np.zeros([wtest,c])
winoutsd = np.zeros([2,c])

# Obtain benchmark target
t = range(10)

# np arrays to save genePools, outputs and fitness
geneArray = np.empty((generations, genes, genomes))
outputArray = np.empty((generations, len(P) - skipstates, genomes)) 
fitnessArray = np.empty((generations, genomes))

# temporary arrays, overwritten each generation
fitnessTemp = np.empty((genomes, fitnessAvg))
# trained_output = np.empty((len(P) - skipstates, fitnessAvg)) !!!!!!!!!!!!!!!!!!
# outputTemp = np.empty((len(P) - skipstates, genomes)) !!!!!!!!!!!!!!!!!!!
controlVoltages = np.zeros(genes+4)

# initialize save directory
saveDirectory = SaveLib.createSaveDirectory(filepath, name)

# initialize main figure
mainFig = PlotBuilder.initMainFigEvolution(genes, generations, genelabels, generange)


# initialize instruments
ivvi = IVVIrack.initInstrument()

for i in range(generations):

    for j in range(genomes):

        # set the DAC voltages
        for k in range(genes-1):
            controlVoltages[k+4] = Evolution.mapGenes(
                generange[k], genePool.pool[k, j])
        IVVIrack.setControlVoltages(ivvi, controlVoltages)

        #set the input scaling
        x_scaled = x * Evolution.mapGenes(generange[-1], genePool.pool[genes-1, j])

        #wait after setting DACs
        time.sleep(1)

        for avgIndex in range(fitnessAvg):

            # feed input to adwin
            for n in range(c):
                inputVoltages = [(win[n,0]-0.5)*2*vin, (win[n,1]-0.5)*2*vin, (win[n,2]-0.5)*2*vin, (win[n,3]-0.5)*2*vin, va,vb]
                print(inputVoltages)
                IVVIrack.setControlVoltages(ivvi, inputVoltages)
                time.sleep(0.2)
                for m in range(wtest):
                    measureddata = np.asarray(nidaqIO.IO(y, fs)) * 10

                    measureddata = np.average(measureddata)
          #   print('measured output current is')
                
                    output[n,m]=np.transpose(measureddata)
    
            # plot genome
            PlotBuilder.currentGenomeEvolution(mainFig, genePool.pool[:, j])
            
            # Train output
            # trained_output[:, avgIndex] =10 * np.asarray(output)  # empty for now, as we have only one output node

            # Calculate fitness
            fitnessTemp[j, avgIndex]= PostProcess.fitnessEvolutionCalssif(output, fitnessParameters)
            for l in range(len(output)):
                y[l] = np.average(output[l])
            #plot output
            PlotBuilder.currentOutputEvolution(mainFig, t, target, output, j + 1, i + 1, fitnessTemp[j, avgIndex])

        outputTemp[:, j] = output[:, np.argmin(fitnessTemp[j, :])]

    genePool.fitness = fitnessTemp.min(1)
    print("Generation nr. " + str(i + 1) + " completed")
    print("Highest fitness: " + str(max(genePool.fitness)))

    # save generation data
    geneArray[i, :, :] = genePool.returnPool()
    outputArray[i, :, :] = y[:, np.argmin(fitnessTemp[j, :])]
    fitnessArray[i, :] = fitnessTemp.min(1)

    PlotBuilder.currentOutputEvolution(mainFig, t, target, output, j + 1, i + 1, fitnessTemp[j, avgIndex])
    PlotBuilder.updateMainFigEvolution(mainFig, geneArray, fitnessArray, outputArray, i + 1, t, target, output)
	
	#save generation
    SaveLib.saveMain(saveDirectory, geneArray, outputArray, fitnessArray, t, x, output)
	
    # evolve the next generation
    genePool.nextGen()

SaveLib.saveMain(filepath, geneArray, outputArray, fitnessArray, t, x, output)

PlotBuilder.finalMain(mainFig)
