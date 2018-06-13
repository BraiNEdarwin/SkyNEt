'''
Builds plots for specific data sets
A function for each type of data that has to be plotted
'''
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import math


def mapGenes(generange, gene):
    return generange[0] + gene * (generange[1] - generange[0])


def genericPlot1D(x, y, xlabel, ylabel, title):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()


def genericPlot2D(x, y, xlabel, ylabel, title):
    length, dim = np.shape(y)
    plt.figure()
    for i in range(dim):
        plt.plot(x, y[:, i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()


def showPlot():
    plt.show()


def bigDaddy(geneArray, fitnessArray):
    generations = geneArray.shape[0]
    genes = geneArray.shape[1]
    bigDaddyArray = np.empty((geneArray.shape[0], genes))
    for i in range(generations):
        bigDaddyArray[i, :] = geneArray[i, :, np.argmax(fitnessArray[i, :])]

    for i in range(genes):
        plt.subplot(genes, 1, i + 1)
        plt.plot(bigDaddyArray[:, i])
    plt.show()


def bigDaddyMain(mainFig, geneArray, fitnessArray, currentGeneration):
    genes = geneArray.shape[1]
    bigDaddyArray = np.empty((currentGeneration, genes))
    for i in range(currentGeneration):
        bigDaddyArray[i, :] = geneArray[i, :, np.argmax(fitnessArray[i, :])]

    for i in range(genes):
        mainFig.axes[i * 2].plot(range(1, currentGeneration + 1),
                             bigDaddyArray[0:currentGeneration, i], 'r-x')

    plt.pause(0.01)


def fitnessMain(mainFig, fitnessArray, currentGeneration):
    mainFig.axes[-2].plot(range(1, currentGeneration + 1),
                          np.max(fitnessArray, 1)[0:currentGeneration], 'r-x')
    plt.pause(0.01)

def fitnessMainEvolution(mainFig, fitnessArray, currentGeneration):
    mainFig.axes[-3].plot(range(1, currentGeneration + 1),
                          np.max(fitnessArray, 1)[0:currentGeneration], 'r-x')
    plt.pause(0.01)


def outputMain(mainFig, t, target, outputArray, fitnessArray, currentGeneration):
    mainFig.axes[-1].lines.clear()
    mainFig.axes[-1].plot(t, outputArray[currentGeneration - 1,
                                         :, np.argmax(fitnessArray[currentGeneration - 1])], 'r')
    mainFig.axes[-1].plot(t, target, 'b--')
    mainFig.axes[-1].legend(['Trained output', 'Target'], loc = 1)
    plt.pause(0.01)

def outputMainEvolution(mainFig, t, target, outputArray, fitnessArray, currentGeneration):
    mainFig.axes[-2].lines.clear()
    mainFig.axes[-2].plot(t, outputArray[currentGeneration - 1,
                                         :, np.argmax(fitnessArray[currentGeneration - 1])], 'r')
    mainFig.axes[-2].plot(t, target, 'b--')
    mainFig.axes[-2].legend(['Best output', 'Target'], loc = 1)
    mainFig.axes[-2].set_title('Best output of last generation / fitness ' + str(np.max(fitnessArray[currentGeneration - 1])) )
    plt.pause(0.01)

def outputMainClassification(mainFig, outputArray, win, fitnessArray, currentGeneration):
    out = outputArray[currentGeneration - 1,np.argmax(fitnessArray[currentGeneration - 1]),:,:]
    y = np.zeros(len(out))
    mainFig.axes[-2].lines.clear()
    for i in range(len(out)):
        y[i] = np.average(out[i])
        a = mainFig.axes[-2].plot(out[i],np.arange(0,len(out[i])),dashes = [1,1], label = win[i])
        mainFig.axes[-2].plot(out[i],np.arange(0,len(out[i])),'x',color = a[0].get_color())
        mainFig.axes[-2].plot([y[i],y[i]],[0,len(out[0])],dashes = [5,5],color = a[0].get_color())
    mainFig.axes[-2].legend(loc = 1)
    mainFig.axes[-2].set_title('Best output of last generation / fitness ' + str(np.max(fitnessArray[currentGeneration - 1])) )
    plt.pause(0.01)

def currentOutputEvolution(mainFig, t, target, currentOutput, genome, currentGeneration, fitness):
    mainFig.axes[-1].lines.clear()
    mainFig.axes[-1].plot(t, currentOutput, 'r')
    mainFig.axes[-1].plot(t, target, 'b--')
    mainFig.axes[-1].legend(['Current output', 'Target'], loc = 1)
    mainFig.axes[-1].set_title('Genome ' + str(genome) + '/Generation ' + str(currentGeneration) + '/fitness ' + '{0:.2f}'.format(fitness))
    plt.pause(0.01)

def currentOutputClassification(mainFig, out, win, genome, currentGeneration, fitness):
    y = np.zeros(len(out))
    mainFig.axes[-1].lines.clear()
    for i in range(len(out)):
        y[i] = np.average(out[i])
        a = mainFig.axes[-1].plot(out[i],np.arange(0,len(out[i])),dashes = [1,1], label = win[i])
        mainFig.axes[-1].plot(out[i],np.arange(0,len(out[i])),'x',color = a[0].get_color())
        mainFig.axes[-1].plot([y[i],y[i]],[0,len(out[0])],dashes = [5,5],color = a[0].get_color())
    mainFig.axes[-1].legend(loc = 1)
    mainFig.axes[-1].set_title('Genome ' + str(genome) + '/Generation ' + str(currentGeneration) + '/fitness ' + '{0:.2f}'.format(fitness))
    plt.pause(0.01)

def currentGenomeEvolution(mainFig, genome):
    mainFig.axes[-4].lines.clear()
    mainFig.axes[-4].plot(range(1, len(genome) + 1), genome, 'r-x')
    plt.pause(0.01)

def statsMain(mainFig, geneArray, currentGeneration):    
	generations = geneArray.shape[0]
	statstext = mainFig.text(0.5, 0.8, 'hello')

def genePoolVisual(genePool):
    plt.matshow(genePool)
    plt.show()


def initMainFig(genes, generations, genelabels, generange):
    # plt.ion()
    plt.ioff()
    mainFig = plt.figure()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.pause(0.01)
    spec = gridspec.GridSpec(ncols=3, nrows=genes)
    # big daddy (i.e. best genom) plots
    for i in range(genes):
        ax = mainFig.add_subplot(spec[i, 0])
        ax.set_xlim(1, generations)
        ax.set_ylim(0, 1)
        ax.grid()
        ax.set_title(genelabels[i])

        twinax = ax.twinx()
        twinax.set_ylim(mapGenes(generange[i], 0), mapGenes(generange[i], 1))
        twinax.tick_params('y', colors='r')

    axFitness = mainFig.add_subplot(
        spec[0:math.floor(genes / 2), 1:])  # fitness history
    axFitness.set_xlim(1, generations)
    axFitness.grid()
    axFitness.set_title('Fitness')

    axOutput = mainFig.add_subplot(
        spec[math.ceil(genes / 2):, 1:])  # current best output
    axOutput.grid()
    axOutput.set_title('Best output of last generation')


    mainFig.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.pause(0.01)
    return mainFig

def initMainFigEvolution(genes, generations, genelabels, generange):
    # plt.ion()
    plt.ioff()
    mainFig = plt.figure()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.pause(0.01)
    spec = gridspec.GridSpec(ncols=3, nrows=genes + 1)
    
    # big daddy (i.e. best genom) plots
    for i in range(genes):
        ax = mainFig.add_subplot(spec[i, 0])
        ax.set_xlim(1, generations)
        ax.set_ylim(0, 1)
        ax.grid()
        ax.set_title(genelabels[i])

        twinax = ax.twinx()
        twinax.set_ylim(mapGenes(generange[i], 0), mapGenes(generange[i], 1))
        twinax.tick_params('y', colors='r')
    
    # current genome plot
    ax = mainFig.add_subplot(spec[genes, 0])
    ax.set_xlim(1, genes)
    ax.set_ylim(0,1)
    ax.grid()
    ax.set_title('Current genome')

    #fitness history plot
    axFitness = mainFig.add_subplot(
        spec[0:math.floor(genes / 3) + 1, 1:])  # fitness history
    axFitness.set_xlim(1, generations)
    axFitness.grid()
    axFitness.set_title('Fitness')

    #best output of last generation
    axOutput = mainFig.add_subplot(
        spec[math.floor(genes / 3) + 1: 2 * math.floor(genes / 3) + 1, 1:])  # current best output
    axOutput.grid()
    axOutput.set_title('Best output of last generation')

    #current output
    axCurrentOutput = mainFig.add_subplot(
        spec[2 * math.floor(genes / 3) + 1: , 1:])  # current best output
    axCurrentOutput.grid()
    axCurrentOutput.set_title('Current genome output')

    mainFig.subplots_adjust(hspace=0.8, wspace=0.8)
    plt.pause(0.01)
    return mainFig


def updateMainFig(mainFig, geneArray, fitnessArray, outputArray, currentGeneration, t, target):
    bigDaddyMain(mainFig, geneArray, fitnessArray, currentGeneration)
    fitnessMain(mainFig, fitnessArray, currentGeneration)
    outputMain(mainFig, t, target, outputArray,
               fitnessArray, currentGeneration)
    #statsMain(mainFig, geneArray, currentGeneration)


def updateMainFigEvolution(mainFig, geneArray, fitnessArray, outputArray, currentGeneration, t, target, currentOutput):
    bigDaddyMain(mainFig, geneArray, fitnessArray, currentGeneration)
    fitnessMainEvolution(mainFig, fitnessArray, currentGeneration)
    outputMainEvolution(mainFig, t, target, outputArray,
               fitnessArray, currentGeneration)
    #currentOutputEvolution(mainFig, t, target, currentOutput)
    #statsMain(mainFig, geneArray, currentGeneration)

def updateMainFigClassification(mainFig, geneArray, fitnessArray, outputArray, currentGeneration, win):
    bigDaddyMain(mainFig, geneArray, fitnessArray, currentGeneration)
    fitnessMainEvolution(mainFig, fitnessArray, currentGeneration)
    outputMainClassification(mainFig, outputArray, win, 
               fitnessArray, currentGeneration)
    #currentOutputEvolution(mainFig, t, target, currentOutput)
    #statsMain(mainFig, geneArray, currentGeneration)

def finalMain(mainFig):
    plt.show()
