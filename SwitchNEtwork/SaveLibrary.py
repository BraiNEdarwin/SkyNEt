'''
Library that handles saving all relevant data.
'''
import numpy as np
import math
import time
import os
from shutil import copyfile

def createSaveDirectory(filepath, name):
	datetime = time.strftime("%d_%m_%Y_%H%M%S")
	filepath = filepath + '_' + datetime + '_' + name
	if not os.path.exists(filepath):
		os.makedirs(filepath)	
	return filepath

def saveMain(filepath, genearray, outputarray, fitnessarray, successarray,timearray):
	saveArrays(filepath, genearray, outputarray, fitnessarray, successarray,timearray)
	saveConfig(filepath)

def saveArrays(filepath, genearray, outputarray, fitnessarray, successarray,timearray):
	np.savez(os.path.join(filepath, 'DataArrays'), genearray=genearray,
		outputarray=outputarray, fitnessarray=fitnessarray, successarray = successarray,timearray = timearray)

def saveConfig(filepath):
	copyfile('setup.txt', os.path.join(filepath, 'setup.txt'))

#IV related

def SaveMainIV(filepath, currentlist):
	saveArraysIV(filepath, currentlist)
	saveConfigIV(filepath)

def saveArraysIV(filepath, currentlist):
	np.savez(os.path.join(filepath, 'IVDataz'), currentlist = currentlist)

def saveConfigIV(filepath):
	copyfile('setup.txt', os.path.join(filepath, 'setup.txt'))

