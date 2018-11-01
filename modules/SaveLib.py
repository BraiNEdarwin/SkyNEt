'''
Library that handles saving all relevant data.
'''
import numpy as np
import math
import time
import os
from shutil import copyfile
import sys

def saveArrays(filepath, **kwargs):
    '''
    Saves an arbitrary amount of numpy arrays in a numpy array archive named
    data. Saves it at the location specified by filepath.
    '''
    if 'filename' in kwargs:
        filename = kwargs['filename']
    else:
        filename = 'data'
    np.savez(os.path.join(filepath, filename), **kwargs)

def copyFiles(filepath):
    '''
    This function copies any .py files in the current directory to the directory
    specified by filepath.
    '''
    filenames = os.listdir()
    for filename in filenames:
        if(os.path.isfile(filename)):
            copyfile(filename, os.path.join(filepath, filename))

def saveExperiment(filepath, **kwargs):
    '''
    This function saves all .py files and any numpy arrays specified by **kwargs
    in the directory specified by filepath.
    '''
    saveArrays(filepath, **kwargs)
    copyFiles(filepath)

def createSaveDirectory(filepath, name):
    '''
    This function checks if there exists a directory filepath+datetime_name.
    If not it will create it and return this path.
    '''
    datetime = time.strftime("%Y_%m_%d_%H%M%S")

    filepath = filepath + datetime + '_' + name

    if not os.path.exists(filepath):
        os.makedirs(filepath)
    return filepath
