''''
Measurement script to perform an evolution experiment of a selected
gate. This will initially be tested on the Heliox (with nidaq) setup.
'''

# Import packages
import modules.ReservoirFull as Reservoir
import modules.PlotBuilder as PlotBuilder
import modules.GenerateInput as GenerateInput
import modules.Evolution_Gauss as Evolution
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
b = [-1, 1, 0]
win = np.array(list(itertools.product(*[a,a,a,a])))
c = len(win)

# win = np.array([[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],[1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1],[-1,0,0,0],[-1,1,0,0]])
wtest = 10
fs = 10000
vin = 500.0
y = np.zeros([200])
winout = np.zeros([wtest,c])
winoutsd = np.zeros([2,c])

# Obtain benchmark target
t = range(10)

# np arrays to save genePools, outputs and fitness
geneArray = np.empty((generations, genes, genomes))
outputArray = np.empty((generations, genomes, c, wtest)) 
fitnessArray = np.empty((generations, genomes))

# temporary arrays, overwritten each generation
fitnessTemp = np.empty((genomes, fitnessAvg))
controlVoltages = np.zeros(7)
output = np.empty((c, wtest))

# initialize save directory
saveDirectory = SaveLib.createSaveDirectory(filepath, name)

# initialize main figure
mainFig = PlotBuilder.initMainFigEvolution(genes, generations, genelabels, generange)

# initialize instruments
ivvi = IVVIrack.initInstrument()

for i in range(generations):

    for j in range(genomes):

        # set the DAC voltages
        a1 = np.zeros(3)
        for k in range(genes):
            a1[k] = Evolution.mapGenes(
                generange[k], genePool.pool[k, j])
        print(a1)
        controlVoltages[4:] = a1[:3]
        IVVIrack.setControlVoltages(ivvi, controlVoltages)
        print("current CV1,CV2,CV3")
        print(controlVoltages[4:])
        print("current Scaling:")
        print(a1[3:])

        #set the input scaling
        # x_scaled = x * Evolution.mapGenes(generange[-1], genePool.pool[genes-1, j])

        #wait after setting DACs

        for avgIndex in range(fitnessAvg):

            # feed input to adwin
            for n in range(c):
                inputVoltages = [(win[n,0])*1000-vin, (win[n,1])*1000-vin, (win[n,2])*1000-vin, (win[n,3])*1000-vin]
                print("current input")
                print(inputVoltages)
                IVVIrack.setControlVoltages(ivvi, inputVoltages)
                time.sleep(0.2)
                for m in range(wtest):
                    measureddata = np.asarray(nidaqIO.IO(y, fs)) * 10
                    output[n,m] = np.average(measureddata)
    
            # plot genome
            PlotBuilder.currentGenomeEvolution(mainFig, genePool.pool[:, j])

            # Calculate fitness
            fitnessTemp[j, avgIndex]= PostProcess.fitnessEvolutionClassif(output, fitnessParameters)

            #plot output
            PlotBuilder.currentOutputClassification(mainFig, output, win, j + 1, i + 1, fitnessTemp[j, avgIndex], marker)
        outputArray[i, j, :, :] = output
    genePool.fitness = fitnessTemp.min(1)
    print("Generation nr. " + str(i + 1) + " completed")
    print("Highest fitness: " + str(max(genePool.fitness)))


    # save generation data
    geneArray[i, :, :] = genePool.returnPool()
    fitnessArray[i, :] = fitnessTemp.min(1)

    PlotBuilder.updateMainFigClassification(mainFig, geneArray, fitnessArray, outputArray, i + 1, win, marker)

	#save generation
    SaveLib.saveMainClassification(saveDirectory, geneArray, outputArray, fitnessArray, win)

    # evolve the next generation
    genePool.nextGen()

SaveLib.saveMainClassification(saveDirectory, geneArray, outputArray, fitnessArray, win)
IVVIrack.setControlVoltages(ivvi,np.zeros(7))
PlotBuilder.finalMain(mainFig)
