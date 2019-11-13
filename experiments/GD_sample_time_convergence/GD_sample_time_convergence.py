# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:00:53 2019

This script is used to measure gradient data with increasing sample time T.
The uncertainty in gradient information should decrease as 1/T, as shown in the
lock-in equations (integrals that are approximated to be 0 for large enough T
decay with a rate of 1/T).

@author: Mark Boon
"""

import SkyNEt.experiments.gradient_descent.config_gradient_accuracy as config
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
t = np.arange(0.0, cf.sampleTime[-1], 1/cf.fs)
outputs = []
IVgrad = np.zeros((cf.sampleTime.shape[0], cf.n, cf.controls))
phases = np.zeros((cf.sampleTime.shape[0], cf.n, cf.controls))
sign = np.zeros((cf.sampleTime.shape[0], cf.n, cf.controls))

# Create DC inputs (for the largest sample time, for lower sample times we can take a smaller part of this array)
inputs = np.array(controls[:,np.newaxis] * np.ones(int(cf.sampleTime[-1] * cf.fs)))

# Add AC part to inputs
inputs += np.sin(2 * np.pi * cf.freq[:,np.newaxis] * t) * cf.waveAmplitude[:,np.newaxis]

# Add ramping up and down to the inputs
ramp = np.ones((cf.controls, 1)) * np.linspace(0, cf.staticControls, int(cf.fs*cf.rampT))    
inputs_ramped = np.concatenate((ramp,inputs,ramp[:,:,:-1]),axis=1)

# Data acquisition loop
for g in range(cf.sampleTime.shape[0]):
    print('Sampling for sample time ' + str(g+1) + '/' + str(cf.sampleTime.shape[0]))
    data = np.zeros((cf.n, int(cf.sampleTime[g] * cf.fs)))
    
    for i in range(cf.n):       
        print('Sampling iteration ' + str(i+1) + '/' + str(cf.n) + '...')
        dataRamped = InstrumentImporter.nidaqIO.IO_cDAQ(inputs_ramped[:, :int(cf.sampleTime[g]*cf.fs)], cf.fs) * cf.gainFactor
        data[i,:] = dataRamped[0, int(cf.fs*cf.rampT):-int(cf.fs*cf.rampT)]   # Cut off the ramping up and down part 
        

    
        # Lock-in technique to determine gradients
        x_ref = np.arange(0.0, cf.sampleTime[g], 1/cf.fs) 
        for j in range(cf.controls): # For each control electrode, calculate the gradient (7 out of 8 times only 1 of the 7 controls is interesting)
            y_ref1 = np.sin(cf.freq[j] * 2.0*np.pi*x_ref)           # Reference signal 1 (phase = 0)
            y_ref2 = np.sin(cf.freq[j] * 2.0*np.pi*x_ref + np.pi/2) # Reference signal 2 (phase = pi/2)
            
            y1_out = y_ref1*(data[i,:] - np.mean(data[i,:]))
            y2_out = y_ref2*(data[i,:] - np.mean(data[i,:]))
             
            amp1 = np.mean(y1_out) 
            amp2 = np.mean(y2_out)
            IVgrad[g,i,j] = 2*np.sqrt(amp1**2 + amp2**2) / cf.waveAmplitude[j] # Amplitude of output wave (nA) / amplitude of input wave (V)
            phases[g,i,j] = np.arctan2(amp2,amp1)*180/np.pi
            sign[g,i,j] = 1 if abs(phases[g,i,j]) < cf.phase_thres else -1
           
    # Add output data from current factor to outputs list           
    outputs += [data]       
    
    print('Saving...')
    SaveLib.saveExperiment(cf.configSrc, saveDirectory,
                             controls = controls,
                             inputs = inputs,
                             output = outputs,
                             freq = cf.freq,
                             fs = cf.fs,
                             waveAmplitude = cf.waveAmplitude,
                             IVgrad = IVgrad,
                             phases = phases,
                             sign = sign)
    
if cf.device == 'chip':
    InstrumentImporter.reset(0, 0)  


