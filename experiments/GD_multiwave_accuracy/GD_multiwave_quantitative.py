# -*- coding: utf-8 -*-

"""
Created on Wed Nov 13 09:29:46 2019

This script is used to determine the accuracy of the gradient information when applying sine waves to all input electrodes simultaneously.
Multiple parameters are varied:
For an input electrode, increasing frequencies are applied to test how accurate the gradient is at higher frequencies.
This is done for every single electrode.
After this, this is also applied for all electrodes simultaneously.
This can be used to compare: 1) how fast can we sample. 2) is it also accurate for an all AC measurement.

Info on most important arrays:

inputs: array([S x C+1 x C x L])]:  
                                    C+1 combinations of wave experiment (C times for the C electrodes, '+1' for applying waves to all electrodes simultaneously)
                                    S different sets of input configurations
                                    C electrodes input dimension
                                    L sample length
                                
data:  array([S x C+1 x L])]:   C+1 combinations of wave experiment (C times for the C electrodes, '+1' for applying waves to all electrodes simultaneously)
                                S different sets of input configurations
                                L sample length

IVgrad: [S x C+1 x C]:          For every electrode (and all simultaneously) and for every CV set, determine gradient w.r.t. each control voltage

@author: Mark Boon
"""
import SkyNEt.experiments.GD_multiwave_accuracy.config_GD_multiwave_quantitative as config
import SkyNEt.modules.SaveLib as SaveLib
import numpy as np
import math
import matplotlib.pyplot as plt

# Initialize config object
cf = config.experiment_config()

if cf.device == 'chip':
    from SkyNEt.instruments import InstrumentImporter
elif cf.device == 'NN':
    from SkyNEt.modules.Nets.staNNet import staNNet
    import torch
    
# Initialize save directory
if cf.device == 'chip': # Don't save for NN bug testing
    saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# Initialize arrays
controls = cf.staticControls

phases = np.zeros((cf.sets, cf.controls + 1, cf.controls)) # iterations x # input cases x # controls 
IVgrad = np.zeros((cf.sets, cf.controls + 1, cf.controls)) # dI_out/dV_in

# If NN is used as proxy, load network
if cf.device == 'NN':  
    net = staNNet(cf.main_dir + cf.NN_name)
    
# Generate inputs and outputs for experiment
t = np.arange(0.0, cf.periods/cf.freq[0], 1/cf.fs)
if cf.device == 'chip':
    inputs = np.zeros((cf.sets, cf.controls+1, cf.controls, t.shape[0] + int(2*cf.fs*cf.rampT)))
elif cf.device == 'NN':
    inputs = np.zeros((cf.sets, cf.controls+1, cf.controls, t.shape[0]))
outputs = np.zeros((cf.sets, cf.controls+1, t.shape[0] ))

for h in range(cf.sets):
    inputs[h,:,:,int(cf.rampT*cf.fs):-int(cf.rampT*cf.fs)] = controls[h,:,np.newaxis] * np.ones((t.shape[0]))[np.newaxis,:] # DC component
    # Add sine waves on top of DC voltages 
    for g in range(cf.controls+1):
        indices = g # Only add AC signal to a single electrode
        if g == 7: indices = [0,1,2,3,4,5,6] # except for the last measurement, now they all obtain an AC signal
        inputs[h,g,indices,int(cf.rampT*cf.fs):-int(cf.rampT*cf.fs)] += np.sin(2 * np.pi * cf.freq[indices,np.newaxis] * t) * cf.waveAmplitude[indices,np.newaxis]        
    # Add ramping up and down to the inputs for device safety
    if cf.device == 'chip':
        ramp = np.ones(cf.controls+1)[:,np.newaxis,np.newaxis]*(controls[h,:,np.newaxis] * np.linspace(0, 1, int(cf.fs*cf.rampT)))    
        inputs[h,:,:,0:int(cf.rampT*cf.fs)] = ramp
        inputs[h,:,:,-int(cf.rampT*cf.fs):] = ramp[:,:,::-1] 
    elif cf.device == 'NN':
        inputs[h,:,:,:] = pre_inputs
    
# Data acquisition loop
print('Estimated time required for experiment: ' + str(np.sum((cf.controls+1)*cf.sets*cf.periods/cf.freq[0])/60 + (cf.controls+1)*cf.sets*(2*cf.rampT + 0.2)/60) + ' minutes (total sample time)')
for h in range(cf.sets):
    print('Sampling for set ' + str(h+1))
    
    for g in range(cf.controls + 1):
        print('Sampling sines for electrode ' + str(g + 1) + ' (8 = all electrodes)')    
            
        if cf.device == 'chip':
            dataRamped = InstrumentImporter.nidaqIO.IO_cDAQ(inputs[h,g], cf.fs) * cf.gainFactor
            outputs[h,g,:] = dataRamped[0, int(cf.fs*cf.rampT):-int(cf.fs*cf.rampT)]   # Cut off the ramping up and down part
        elif cf.device == 'NN':
            outputs[h,g,:] = net.outputs(torch.from_numpy(inputs[h,g,:,0:int(sampleTime*cf.fs)].T).to(torch.float)) + np.random.normal(0,0.5,data[g,i,:].shape[0])     
            
        # Lock-in technique to determine gradients              
        IVgrad[h,g,:], phases[h,g,:] = cf.lock_in_gradient(outputs[h,g,:], cf.freq, cf.waveAmplitude, cf.fs, cf.phase_thres)
            
    
    if (cf.device == 'chip') and ((h+1)%10==0): # We don't want to save every time we use the NN to test for bugs (save after every 10 sets)
        print('Saving...')
        SaveLib.saveExperiment(cf.configSrc, saveDirectory,
                                 controls = controls,
                                 inputs = inputs,
                                 output = outputs,
                                 sets = cf.sets,
                                 fs = cf.fs,
                                 freq = cf.freq,
                                 waveAmplitude = cf.waveAmplitude,
                                 IVgrad = IVgrad,
                                 phases = phases)   



if cf.device == 'chip':
    InstrumentImporter.reset(0, 0)  