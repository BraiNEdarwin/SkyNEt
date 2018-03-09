'''
Builds plots for specific data sets
A function for each type of data that has to be plotted
'''
import matplotlib.pyplot as plt
import numpy as np


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
		bigDaddyArray[i,:] = geneArray[i, :, np.argmax(fitnessArray[i, :])]
		
	for i in range(genes):
		plt.subplot(genes, 1, i + 1)
		plt.plot(bigDaddyArray[:,i])
	plt.show()
