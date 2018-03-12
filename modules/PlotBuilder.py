'''
Builds plots for specific data sets
A function for each type of data that has to be plotted
'''
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import math

plt.ion()


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
        mainFig.axes[i].plot(range(1, currentGeneration + 1),
                             bigDaddyArray[0:currentGeneration, i], 'r-x')

    plt.pause(0.01)


def fitnessMain(mainFig, fitnessArray, currentGeneration):
    mainFig.axes[-2].plot(range(1, currentGeneration + 1),
                          np.max(fitnessArray, 1)[0:currentGeneration], 'r-x')
    plt.pause(0.01)


def outputMain(mainFig, t, target, outputArray, fitnessArray, currentGeneration):
    mainFig.axes[-1].clear()
    mainFig.axes[-1].grid()
    mainFig.axes[-1].plot(t, outputArray[currentGeneration - 1,
                                         :, np.argmax(fitnessArray[currentGeneration - 1])], 'r')
    mainFig.axes[-1].plot(t, target, 'b--')
    plt.pause(0.01)


def genePoolVisual(genePool):
    plt.matshow(genePool)
    plt.show()


def initMainFig(genes, generations):
    # plt.ion()
    plt.ioff()
    mainFig = plt.figure()
    plt.pause(0.01)
    spec = gridspec.GridSpec(ncols=3, nrows=genes)
    # big daddy (i.e. best genom) plots
    for i in range(genes):
        ax = mainFig.add_subplot(spec[i, 0])
        ax.set_xlim(1, generations)
        ax.set_ylim(0, 1)
        ax.grid()

    axFitness = mainFig.add_subplot(
        spec[0:math.floor(genes / 2), 1:])  # fitness history
    axFitness.set_xlim(1, generations)
    axFitness.grid()

    axOutput = mainFig.add_subplot(
        spec[math.ceil(genes / 2):, 1:])  # current best output
    axOutput.grid()

    mainFig.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.pause(0.01)
    return mainFig


def updateMainFig(mainFig, geneArray, fitnessArray, outputArray, currentGeneration, t, target):
    bigDaddyMain(mainFig, geneArray, fitnessArray, currentGeneration)
    fitnessMain(mainFig, fitnessArray, currentGeneration)
    outputMain(mainFig, t, target, outputArray,
               fitnessArray, currentGeneration)


def finalMain(mainFig):
    plt.show()
