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

inputs:[F x C+1 x C x L]:  
                                    C+1 combinations of wave experiment (C times for the C electrodes, '+1' for applying waves to all electrodes simultaneously)
                                    F different wave factors (to determine the maximum wave frequency while still obtaining accurate data)
                                    C electrodes input dimension
                                    L sample length (differs per factor, if L is lower than the max, it is appended with zeros to fit in the array)
                                
data: [F x C+1 x N x L]:   C+1 combinations of wave experiment (C times for the C electrodes, '+1' for applying waves to all electrodes simultaneously)
                                    F different wave factors (to determine the maximum wave frequency while still obtaining accurate data)
                                    N amount of iterations (for some statistics)
                                    L sample length

IVgrad: [F x C+1 x N x C]:      For every electrode (and all simultaneously) and for every factor, sample N times and determine gradient w.r.t. each control voltage

@author: Mark Boon
"""
import SkyNEt.experiments.GD_multiwave_accuracy.config_GD_multiwave_accuracy as config
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
controls = cf.staticControls*np.ones(cf.controls) # the DC values of the inputs (which are kept constant for the experiment)

phases = np.zeros((cf.factors.shape[0], cf.controls + 1, cf.n, cf.controls)) # iterations x # input cases x # controls 
IVgrad = np.zeros((cf.factors.shape[0], cf.controls + 1, cf.n, cf.controls)) # dI_out/dV_in

# If NN is used as proxy, load network
if cf.device == 'NN':  
    net = staNNet(cf.main_dir + cf.NN_name)
    
# Generate inputs and outputs for experiment
# Since the sample time decreases as the factor for the sample frequencies increases,
# The size of the arrays for different F will vary. Therefore, we use a list which contains
# arrays for the different values for F.
inputs = np.zeros((cf.factors.shape[0], cf.controls+1, cf.controls, int(cf.fs*(2*cf.rampT + cf.periods/(cf.factors[0]*cf.freq[0])))  ))
outputs = np.zeros((cf.factors.shape[0], cf.controls+1, cf.n, int(cf.fs*cf.periods/(cf.factors[0]*cf.freq[0]))))

for h in range(cf.factors.shape[0]):
    freq = cf.freq * cf.factors[h]
    sampleTime = cf.periods/freq[0] # Sample time is dependent on the lowest frequency of the current factor and is always a specific amount of periods of the slowest frequency of the input waves
    t = np.arange(0.0, sampleTime, 1/cf.fs)
    
    pre_inputs = np.ones(cf.controls+1)[:,np.newaxis,np.newaxis]*(np.array(controls[:,np.newaxis] * np.ones(int(sampleTime * cf.fs)) ))[np.newaxis,:] # DC component [C+1, C, L]
    # Add sine waves on top of DC voltages 
    for g in range(cf.controls+1):
        indices = g # Only add AC signal to a single electrode
        if g == 7: indices = [0,1,2,3,4,5,6] # except for the last measurement, now they all obtain an AC signal
        pre_inputs[g,indices,:] += np.sin(2 * np.pi * freq[indices,np.newaxis] * t[:pre_inputs.shape[2]]) * cf.waveAmplitude[indices,np.newaxis]
        
    # Add ramping up and down to the inputs for device safety
    if cf.device == 'chip':
        ramp = np.ones(cf.controls+1)[:,np.newaxis,np.newaxis]*(cf.staticControls[:,np.newaxis] * np.linspace(0, 1, int(cf.fs*cf.rampT)))    
        inputs_ramped = np.concatenate((ramp,pre_inputs,ramp[:,:,::-1]),axis=2)    
        inputs[h,:,:,0:int(cf.fs*(2*cf.rampT + sampleTime))] = inputs_ramped # Now we have [F, C+1, C, L+ramptime]
    elif cf.device == 'NN':
        inputs[h,:,:,0:int(sampleTime*cf.fs)] = pre_inputs

        
# Data acquisition loop
print('Estimated time required for experiment: ' + str(np.sum((cf.controls+1)*cf.n*cf.periods/(cf.freq[0]*cf.factors))/60 + (cf.controls+1)*cf.n*cf.factors.shape[0]*(2*cf.rampT + 0.2)/60) + ' minutes (total sample time)')
for h in range(cf.factors.shape[0]):
    print('Sampling for factor ' + str(cf.factors[h]))
    freq = cf.freq * cf.factors[h]
    sampleTime = cf.periods/freq[0] # Sample time is dependent on the lowest frequency of the current factor and is always a specific amount of periods of the slowest frequency of the input waves   
    
    for g in range(cf.controls + 1):
        print('Sampling sines for electrode ' + str(g + 1) + ' (8 = all electrodes)')
        for i in range(cf.n):
            print('Iteration ' + str(i+1) + '/' + str(cf.n) + '...')        
            
            if cf.device == 'chip':
                dataRamped = InstrumentImporter.nidaqIO.IO_cDAQ(inputs[h,g,:,0:int(cf.fs*(2*cf.rampT + sampleTime))], cf.fs) * cf.gainFactor
                outputs[h,g,i,0:int(sampleTime*cf.fs)] = dataRamped[0, int(cf.fs*cf.rampT):-int(cf.fs*cf.rampT)]   # Cut off the ramping up and down part
            elif cf.device == 'NN':
                outputs[h,g,i,0:int(sampleTime*cf.fs)] = net.outputs(torch.from_numpy(inputs[h,g,:,0:int(sampleTime*cf.fs)].T).to(torch.float)) + np.random.normal(0,0.5,data[g,i,:].shape[0])     
            
            # Lock-in technique to determine gradients              
            IVgrad[h,g,i,:], phases[h,g,i,:] = cf.lock_in_gradient(outputs[h,g,i,0:int(sampleTime*cf.fs)], freq, cf.waveAmplitude, cf.fs, cf.phase_thres)
            
    
    if cf.device == 'chip': # We don't want to save every time we use the NN to test for bugs
        print('Saving...')
        SaveLib.saveExperiment(cf.configSrc, saveDirectory,
                                 controls = controls,
                                 inputs = inputs,
                                 output = outputs,
                                 factors = cf.factors,
                                 fs = cf.fs,
                                 waveAmplitude = cf.waveAmplitude,
                                 IVgrad = IVgrad,
                                 phases = phases)   

if cf.verbose:
    IVmeans = np.mean(IVgrad,axis=2)
    IVstds = np.std(IVgrad,axis=2)
    for i in range(7):
        plt.figure(3)
        plt.errorbar(cf.factors, IVmeans[:,i,i], yerr = IVstds[:,i,i], marker=".", ls='',capsize = 2)
        plt.plot(cf.factors, IVmeans[:,7,i],'k')
        plt.plot(cf.factors, IVmeans[:,7,i] + IVstds[:,7,i],'k--')
        plt.plot(cf.factors, IVmeans[:,7,i] - IVstds[:,7,i],'k--')
        plt.figure(4)
        plt.plot(cf.factors, (1-IVmeans[:,i,i]/IVmeans[:,7,i])*100)
        
        plt.figure(3)
        
        plt.xlabel('factors')
        plt.ylabel('mean gradient')
        
        plt.figure(4)
        plt.xlabel('factors (lowest freq)')
        plt.ylabel('difference single electrode vs all electrodes (%)')
    plt.show()

if cf.device == 'chip':
    InstrumentImporter.reset(0, 0)  