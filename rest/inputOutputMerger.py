# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 15:54:33 2019

Script that merges the input and output data from the wave search such that it
can be used for DataHandler.py and then TrainNets.py

@author: Mark
"""
import numpy as np
import SkyNEt.modules.SaveLib as SaveLib


maindir = r'C:\Users\User\APH\Thesis\Data\wave_search\2019_01_25_221015_wave_search_f2sampleTime_86040s_loadEvery_50s\\'
filenameInput = r'inputData_f2_24h_500Hz.npz'
filenameOutput = r'training_NN_data.npz'


outputs = np.load(maindir + filenameOutput)['output'].transpose()
inputs = np.load(maindir + filenameInput)['waves'].transpose()[0:outputs.shape[0],:]


cropping_mask = np.abs(outputs) < 36.6
outputs = outputs[cropping_mask[:]]
inputs = inputs[cropping_mask[:,0],:]
var_output = np.zeros((outputs.shape[0],1))

#saveArrays(r'C:\Users\User\APH\Thesis\Data\wave_search\2019_01_25_221015_wave_search_f2sampleTime_86040s_loadEvery_50s\\', filename="data_for_training",inputs = inputs, outputs = outputs, var_output = var_output)
