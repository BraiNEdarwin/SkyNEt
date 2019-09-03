# -*- coding: utf-8 -*-

"""
Created on Wed Jul 47 09:29:46 2019

This script is used to determine the accuracy of the gradient information when applying sine waves to all input electrodes simultaneously.
Multiple parameters are varied:
For an input electrode, increasing frequencies are applied to test how accurate the gradient is at higher frequencies.
This is done for every single electrode.
After this, this is also applied for all electrodes simultaneously.
This can be used to compare: 1) how fast can we sample. 2) is it also accurate for an all AC measurement.

Info on most important arrays:

inputs: [C+1 x F x C x L]:  C+1 combinations of wave experiment (C times for the C electrodes, '+1' for applying waves to all electrodes simultaneously)
                                F different wave factors (to determine the maximum wave frequency while still obtaining accurate data)
                                C electrodes input dimension
                                L sample length
                                
data: [C+1 x F x N x L]:        C+1 combinations of wave experiment (C times for the C electrodes, '+1' for applying waves to all electrodes simultaneously)
                                F different wave factors (to determine the maximum wave frequency while still obtaining accurate data)
                                N amount of iterations (for some statistics)
                                L sample length

IVgrad: [C+1 x F x N x C]:      For every electrode (and all simultaneously) and for every factor, sample N times and determine gradient w.r.t. each control voltage

@author: Mark Boon
"""
import SkyNEt.experiments.gradient_descent.config_gradient_accuracy as config
import SkyNEt.modules.SaveLib as SaveLib
from SkyNEt.modules.Nets.staNNet import staNNet
import numpy as np

# Initialize config object
cf = config.experiment_config()

if cf.device == 'chip':
    from SkyNEt.instruments import InstrumentImporter
elif cf.device == 'NN':
    import torch

# Initialize save directory
saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# Initialize arrays
controls = cf.staticControls*np.ones(cf.controls) # the DC values of the inputs (which are kept constant for the experiment)

inputs = np.zeros((cf.controls + 1, cf.factors.shape[0], cf.controls, cf.signallength * cf.fs + int(2 * cf.fs * cf.rampT))) #

data = np.zeros((cf.controls + 1, cf.factors.shape[0], cf.n, cf.signallength * cf.fs)) # cf.controls +1 is because we first experiment with all electrodes taken separately and finally when applying all of them at once.
phases = np.zeros((cf.controls + 1, cf.factors.shape[0], cf.n, cf.controls)) # iterations x # input cases x # controls 
IVgrad = np.zeros((cf.controls + 1, cf.factors.shape[0], cf.n, cf.controls)) # dI_out/dV_in
sign = np.zeros((cf.controls + 1, cf.factors.shape[0], cf.n, cf.controls)) # Sign of the outputting wave form (+1 or -1 depending on the phase)

t = np.arange(0.0, cf.signallength, 1/cf.fs)

# If NN is used as proxy, load network
if cf.device == 'NN':  
    net = staNNet(cf.main_dir + cf.NN_name)

# Main aqcuisition loop

print('Estimated time required for experiment: ' + str((cf.controls+1)*cf.factors.shape[0]*cf.n*cf.signallength/60) + ' minutes (total sample time)')
for g in range(cf.factors.shape[0]): # For every speed factor
  freq = cf.freq *cf.factors[g] # Define the frequency to sample with
  print('sampling with frequencies: ' + str(freq))

  for h in range(cf.controls + 1): # For every single control electrode (+ for all electrodes at once)
    print('Measuring for experiment ' + str(h+1) + '/' + str(cf.controls + 1))
    for i in range(cf.n): 
        print('Iteration ' + str(i+1))
        # Apply DC control voltages:
        inputs[h, g, :, int(cf.fs*cf.rampT):-int(cf.fs*cf.rampT)] = controls[:,np.newaxis] * np.ones(cf.signallength * cf.fs) 
        
        # Add sine waves on top of DC voltages 
        indices = h # Only add AC signal to a single electrode
        if h == 7: indices = [0,1,2,3,4,5,6] # except for the last measurement, now they all obtain an AC signal

        inputs[h, g, indices, int(cf.fs*cf.rampT):-int(cf.fs*cf.rampT)] = inputs[h, g, indices, int(cf.fs*cf.rampT):-int(cf.fs*cf.rampT)] + np.sin(2 * np.pi * freq[indices,np.newaxis] * t) * cf.waveAmplitude[indices,np.newaxis]
        
        # Add ramping up and ramping down the voltages at start and end of iteration (if measuring on real device)
        if cf.device == 'chip':
            for j in range(cf.controls):
                inputs[h, g, j, 0:int(cf.fs*cf.rampT)] = np.linspace(0, inputs[h, g, j, int(cf.fs*cf.rampT)], int(cf.fs*cf.rampT))
                inputs[h, g, j, -int(cf.fs*cf.rampT):] = np.linspace(inputs[h, g, j, -int(cf.fs*cf.rampT + 1)], 0, int(cf.fs*cf.rampT))    
            
        # Measure output
        if cf.device == 'chip':
            dataRamped = InstrumentImporter.nidaqIO.IO_cDAQ(inputs[h,g,:,:], cf.fs) * cf.gainFactor
            data[h,g,i,:] = dataRamped[0, int(cf.fs*cf.rampT):-int(cf.fs*cf.rampT)]   # Cut off the ramping up and down part
        elif cf.device == 'NN':
            data[h,g,i,:] = net.outputs(torch.from_numpy(inputs[h,g,:,int(cf.fs*cf.rampT):-int(cf.fs*cf.rampT)].T).to(torch.float))

        
        # Lock-in technique to determine gradients
        x_ref = np.arange(0.0, cf.signallength, 1/cf.fs)
            

        for j in range(cf.controls):
          y_ref1 = np.sin(freq[j] * 2.0*np.pi*x_ref)          # Reference signal 1 (phase = 0)
          y_ref2 = np.sin(freq[j] * 2.0*np.pi*x_ref + np.pi/2) # Reference signal 2 (phase = pi/2)
            
          y1_out = y_ref1*data[h,g,i,:] # - np.mean(data[h,i,:]))
          y2_out = y_ref2*data[h,g,i,:] # - np.mean(data[h,i,:]))

          ################################################################################################################################
          # To make a fair comparison for the speed test, we keep the amount of wavelengths in one sample constant and therefore
          # artifically only take output data up to signallength/factor to determine IVgrad
          ################################################################################################################################
                
          amp1 = np.mean(y1_out[:cf.signallength * cf.fs // cf.factors[g]]) 
          amp2 = np.mean(y2_out[:cf.signallength * cf.fs // cf.factors[g]])
          IVgrad[h,g,i,j] = 2*np.sqrt(amp1**2 + amp2**2) / cf.waveAmplitude[j] # Amplitude of output wave (nA) / amplitude of input wave (V)
          phases[h,g,i,j] = np.arctan2(amp2,amp1)*180/np.pi
          sign[h,g,i,j] = 1 if abs(phases[h,g,i,j]) < cf.phase_thres else -1
          
  print('Saving...')
  SaveLib.saveExperiment(cf.configSrc, saveDirectory,
                         controls = controls,
                         inputs = inputs,
                         output = data,
                         factors = cf.factors,
                         fs = cf.fs,
                         t = t,
                         signallength = cf.signallength,
                         waveAmplitude = cf.waveAmplitude,
                         IVgrad = IVgrad,
                         sign=sign)
    

if cf.device == 'chip':
    InstrumentImporter.reset(0, 0)  