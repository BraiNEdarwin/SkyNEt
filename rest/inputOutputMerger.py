# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 15:54:33 2019

Script that processes the data from wave_search such that it can be used for 
TrainNets.py. It adds 

@author: Mark
"""
import numpy as np
import SkyNEt.modules.SaveLib as SaveLib


maindir = r'C:\Users\User\APH\Thesis\Data\wave_search\2019_01_30_123621_characterization_20h_batch_25s_fs_500_f_2\\'
filename = 'training_NN_data.npz'


outputs = np.load(maindir + filename)['output'].transpose()
inputs = np.linspace(0, outputs.shape[0] - 1, outputs.shape[0], dtype = int)[:,np.newaxis]


cropping_mask = np.abs(outputs) < 36
outputs = outputs[cropping_mask[:]]
inputs = inputs[cropping_mask[:,0],:]
var_output = np.zeros((outputs.shape[0],1))

SaveLib.saveArrays(maindir, filename="data_for_training",inputs = inputs, outputs = outputs, var_output = var_output)
