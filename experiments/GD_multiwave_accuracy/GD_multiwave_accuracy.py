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

inputs:[F x array([C+1 x C x L])]:  Since L differs per factor F, we cannot make an array, but rather use a list of arrays.
                                    C+1 combinations of wave experiment (C times for the C electrodes, '+1' for applying waves to all electrodes simultaneously)
                                    F different wave factors (to determine the maximum wave frequency while still obtaining accurate data)
                                    C electrodes input dimension
                                    L sample length
                                
data: [F x array([C+1 x N x L])]:   C+1 combinations of wave experiment (C times for the C electrodes, '+1' for applying waves to all electrodes simultaneously)
                                    F different wave factors (to determine the maximum wave frequency while still obtaining accurate data)
                                    N amount of iterations (for some statistics)
                                    L sample length

IVgrad: [F x C+1 x N x C]:      For every electrode (and all simultaneously) and for every factor, sample N times and determine gradient w.r.t. each control voltage

@author: Mark Boon
"""
import SkyNEt.experiments.GD_multiwave_accuracy.config_GD_multiwave_accuracy as config
import SkyNEt.modules.SaveLib as SaveLib
from SkyNEt.instruments import InstrumentImporter
import numpy as np
import math

# Initialize config object
cf = config.experiment_config()

# Initialize save directory
saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# Initialize arrays
controls = cf.staticControls*np.ones(cf.controls) # the DC values of the inputs (which are kept constant for the experiment)

phases = np.zeros((cf.factors.shape[0], cf.controls + 1, cf.n, cf.controls)) # iterations x # input cases x # controls 
IVgrad = np.zeros((cf.factors.shape[0], cf.controls + 1, cf.n, cf.controls)) # dI_out/dV_in
sign = np.zeros((cf.factors.shape[0], cf.controls + 1, cf.n, cf.controls)) # Sign of the outputting wave form (+1 or -1 depending on the phase)

# Generate inputs and outputs for experiment
# Since the sample time decreases as the factor for the sample frequencies increases,
# The size of the arrays for different F will vary. Therefore, we use a list which contains
# arrays for the different values for F.
inputs = []
outputs = []

for h in range(cf.factors.shape[0]):
    freq = cf.freq * cf.factors[h]
    sampleTime = cf.periods*math.ceil(1/freq[0]) # Sample time is dependent on the lowest frequency of the current factor and is always a specific amount of periods of the slowest frequency of the input waves
    t = np.arange(0.0, sampleTime, 1/cf.fs)
    
    pre_inputs = np.ones(cf.controls+1)[:,np.newaxis,np.newaxis]*(np.array(controls[:,np.newaxis] * np.ones(sampleTime * cf.fs)))[np.newaxis,:] # DC component [C+1, C, L]
    # Add sine waves on top of DC voltages 
    for g in range(cf.controls+1):
        indices = g # Only add AC signal to a single electrode
        if g == 7: indices = [0,1,2,3,4,5,6] # except for the last measurement, now they all obtain an AC signal
        pre_inputs[g,indices,:] += np.sin(2 * np.pi * freq[indices,np.newaxis] * t) * cf.waveAmplitude[indices,np.newaxis]
        
    # Add ramping up and down to the inputs
    ramp = np.ones(cf.controls+1)[:,np.newaxis,np.newaxis]*(np.ones((cf.controls, 1)) * np.linspace(0, cf.staticControls, int(cf.fs*cf.rampT)))    
    inputs_ramped = np.concatenate((ramp,pre_inputs,ramp[:,:,:-1]),axis=2)
    
    inputs += [inputs_ramped] # Now we have [F, C+1, C, L+ramptime]

# Data acquisition loop
print('Estimated time required for experiment: ' + str(np.sum((cf.controls+1)*cf.factors.shape[0]*cf.n*cf.periods/(cf.freq[0]*cf.factors))/60) + ' minutes (total sample time)')
for h in range(cf.factors.shape[0]):
    print('Sampling for factor ' + str(cf.factors[h]))
    freq = cf.freq * cf.factors[h]
    sampleTime = cf.periods*math.ceil(1/freq[0]) # Sample time is dependent on the lowest frequency of the current factor and is always a specific amount of periods of the slowest frequency of the input waves   
    data = np.zeros((cf.controls+1, cf.n, sampleTime)) # Create output data array per factor, since each factor has its own sample time (see inputs)  
    
    for g in range(cf.controls + 1):
        print('Sampling sines for electrode ' + str(g + 1) + ' (8 = all elecrodes)')
        for i in range(cf.n):
            print('Iteration ' + str(i+1) + '/' + str(cf.n) + '...')        
            dataRamped = InstrumentImporter.nidaqIO.IO_cDAQ(inputs[h,g,:,:], cf.fs) * cf.gainFactor
            data[g,i,:] = dataRamped[0, int(cf.fs*cf.rampT):-int(cf.fs*cf.rampT)]   # Cut off the ramping up and down part
            
            # Lock-in technique to determine gradients
            x_ref = np.arange(0.0, sampleTime, 1/cf.fs)
            freq = cf.freq * cf.factors[h]    
            for j in range(cf.controls): # For each control electrode, calculate the gradient (7 out of 8 times only 1 of the 7 controls is interesting)
              y_ref1 = np.sin(freq[j] * 2.0*np.pi*x_ref)           # Reference signal 1 (phase = 0)
              y_ref2 = np.sin(freq[j] * 2.0*np.pi*x_ref + np.pi/2) # Reference signal 2 (phase = pi/2)
                
              y1_out = y_ref1*(data[g,i,:] - np.mean(data[g,i,:]))
              y2_out = y_ref2*(data[g,i,:] - np.mean(data[g,i,:]))
                 
              amp1 = np.mean(y1_out) 
              amp2 = np.mean(y2_out)
              IVgrad[h,g,i,j] = 2*np.sqrt(amp1**2 + amp2**2) / cf.waveAmplitude[j] # Amplitude of output wave (nA) / amplitude of input wave (V)
              phases[h,g,i,j] = np.arctan2(amp2,amp1)*180/np.pi
              sign[h,g,i,j] = 1 if abs(phases[h,g,i,j]) < cf.phase_thres else -1
              
    # Add output data from current factor to outputs list           
    outputs += [data]       
    print('Saving...')
    SaveLib.saveExperiment(cf.configSrc, saveDirectory,
                             controls = controls,
                             inputs = inputs,
                             output = outputs,
                             factors = cf.factors,
                             fs = cf.fs,
                             waveAmplitude = cf.waveAmplitude,
                             IVgrad = IVgrad,
                             phases = phases,
                             sign = sign)
    
if cf.device == 'chip':
    InstrumentImporter.reset(0, 0)  