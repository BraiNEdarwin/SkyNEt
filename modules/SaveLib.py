'''
Library that handles saving all relevant data.
'''
import numpy as np
import math
import time
import os
from shutil import copyfile


def saveArrays(filepath, geneArray, outputArray, fitnessArray, t, inp, outp):
    np.savez(os.path.join(filepath, 'nparrays'), geneArray=geneArray,
             outputArray=outputArray, fitnessArray=fitnessArray, t=t, inp=inp, outp=outp)

def saveArraysClassification(filepath, geneArray, outputArray, fitnessArray, win):
    np.savez(os.path.join(filepath, 'nparrays'), geneArray=geneArray,
             outputArray=outputArray, fitnessArray=fitnessArray, win=win)

def saveConfig(filepath):
    copyfile('config.txt', os.path.join(filepath, 'config.txt'))


def saveMain(filepath, geneArray, outputArray, fitnessArray, t, inp, outp):
    saveArrays(filepath, geneArray, outputArray, fitnessArray, t, inp, outp)
    saveConfig(filepath)

def saveMainClassification(filepath, geneArray, outputArray, fitnessArray, win):
    saveArraysClassification(filepath, geneArray, outputArray, fitnessArray, win)
    saveConfig(filepath)

	
def createSaveDirectory(filepath, name):
    datetime = time.strftime("%d_%m_%Y_%H%M%S")
    filepath = filepath + '\\' + datetime + '_' + name
    if not os.path.exists(filepath):
        os.makedirs(filepath)	
    return filepath