# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:00:53 2019

This script is used to measure gradient data with increasing sample time T.
The uncertainty in gradient information should decrease as 1/T, as shown in the
lock-in equations (integrals that are approximated to be 0 for large enough T
decay with a rate of 1/T).

outputs [T x [N X L]]:  T amount of sample times
                        N repetitions of experiment (to compute std)
                        L data points
       
IVgrad [T x N x C]:     T amount of sample times
                        N repetitions of experiment
                        C controls to compute gradient for

IVmeans [T x C]:        T amount of sample times
                        C controls to compute gradient for

@author: Mark Boon
"""

import SkyNEt.experiments.GD_sample_time_convergence.config_GD_sample_time_convergence as config
import SkyNEt.modules.SaveLib as SaveLib
from SkyNEt.modules.Nets.staNNet import staNNet
import numpy as np
import matplotlib.pyplot as plt

# Initialize config object
cf = config.experiment_config()

if cf.device == 'chip':
    from SkyNEt.instruments import InstrumentImporter
elif cf.device == 'NN':
    import torch
    
# Initialize save directory
if cf.device == 'chip': # Don't save for NN bug testing
    saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# Initialize arrays
controls = cf.staticControls*np.ones(cf.controls) # the DC values of the inputs (which are kept constant for the experiment)
t = np.arange(0.0, cf.sampleTime[-1], 1/cf.fs)
outputs = []
IVgrad = np.zeros((cf.sampleTime.shape[0], cf.n, cf.controls))
phases = np.zeros((cf.sampleTime.shape[0], cf.n, cf.controls))

# If NN is used as proxy, load network
if cf.device == 'NN':  
    net = staNNet(cf.main_dir + cf.NN_name)

# Create DC inputs (for the largest sample time, for lower sample times we can take a smaller part of this array)
inputs = np.array(controls[:,np.newaxis] * np.ones(int(cf.sampleTime[-1] * cf.fs)))

# Add AC part to inputs
inputs += np.sin(2 * np.pi * cf.freq[:,np.newaxis] * t) * cf.waveAmplitude[:,np.newaxis]

# Add ramping up and down to the inputs
ramp = cf.staticControls[:,np.newaxis] * np.linspace(0, 1, int(cf.fs*cf.rampT))    

# Data acquisition loop
for g in range(cf.sampleTime.shape[0]):
    print('Sampling for sample time ' + str(g+1) + '/' + str(cf.sampleTime.shape[0]))
    data = np.zeros((cf.n, int(cf.sampleTime[g] * cf.fs)))
    
    for i in range(cf.n):       
        print('Sampling iteration ' + str(i+1) + '/' + str(cf.n) + '...')
        if cf.device == 'chip':
            dataRamped = InstrumentImporter.nidaqIO.IO_cDAQ(np.concatenate((ramp,inputs[:,:int(cf.sampleTime[g]*cf.fs)],ramp[:,:,:-1]),axis=1), cf.fs) * cf.gainFactor
            data[i,:] = dataRamped[0, int(cf.fs*cf.rampT):-int(cf.fs*cf.rampT)]   # Cut off the ramping up and down part 
        elif cf.device == 'NN':
            data[i,:] = net.outputs(torch.from_numpy(inputs[:,:int(cf.sampleTime[g]*cf.fs)].T).to(torch.float)) + np.random.normal(0,0.5,data[i,:].shape[0])     
            
        # Lock-in technique to determine gradients        
        IVgrad[g,i,:], phases[g,i,:] = cf.lock_in_gradient(data[i,:], cf.freq, cf.waveAmplitude, cf.fs, cf.phase_thres)
 
    # Add output data from current factor to outputs list           
    outputs += [data]       
    
    if cf.device == 'chip':
        print('Saving...')
        SaveLib.saveExperiment(cf.configSrc, saveDirectory,
                                 controls = controls,
                                 inputs = inputs,
                                 output = outputs,
                                 freq = cf.freq,
                                 fs = cf.fs,
                                 waveAmplitude = cf.waveAmplitude,
                                 IVgrad = IVgrad,
                                 phases = phases)
        
if cf.verbose:
    IVmeans = np.mean(IVgrad,axis=1)
    IVstds = np.std(IVgrad,axis=1)  
    for i in range(cf.controls):
        plt.figure()
        plt.errorbar(cf.sampleTime, IVmeans[:,i], yerr = IVstds[:,i], marker=".", ls='',capsize = 2)
       
if cf.device == 'chip':
    InstrumentImporter.reset(0, 0)  


