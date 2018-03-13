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


def saveConfig(filepath):
    copyfile('config.txt', os.path.join(filepath, 'config.txt'))


def saveMain(filepath, geneArray, outputArray, fitnessArray, t, inp, outp):
    datetime = time.strftime("%d_%m_%Y_%H%M")
    filepath = filepath + '_' + datetime
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    saveArrays(filepath, geneArray, outputArray, fitnessArray, t, inp, outp)
    saveConfig(filepath)
