# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 08:29:46 2018

This script applies the SPSA algorithm on the device.

@author: Mark Boon
"""

import SkyNEt.experiments.spsa.config_spsa as config
import SkyNEt.modules.SaveLib as SaveLib
import SkyNEt.modules.PlotBuilder as PlotBuilder
from SkyNEt.modules.Nets.lightNNet import lightNNet

import time
import numpy as np
import random

# Initialize config object
cf = config.experiment_config()

if cf.device == 'chip':
    from SkyNEt.instruments.DAC import IVVIrack
    from SkyNEt.instruments import InstrumentImporter
elif cf.device == 'NN':
    import torch
    
# Initialize input and target
t = cf.InputGen()[0]  # Time array
x = np.asarray(cf.InputGen()[1:3])  # Array with P and Q signal
w = cf.InputGen()[3]  # Weight array
target = cf.gainFactor * cf.targetGen()[1]  # Target signal

# Initialize at arrays
yplus = np.zeros(cf.n)
yminus = np.zeros(cf.n)

theta = np.zeros((cf.n + 1, cf.controls))   # Control voltages
thetaplus = np.zeros((cf.n, cf.controls))
thetaminus = np.zeros((cf.n, cf.controls))
delta = np.zeros((cf.n, cf.controls))       # Direction of optimization step
ghat = np.zeros((cf.n, cf.controls))        # Changes in the controls
outputminus = np.zeros((cf.n, x.shape[1]))  # output of thetaplus as control
outputplus = np.zeros((cf.n, x.shape[1]))   # output of thetaminus as control
# Initialize controls at a random point
theta[0,:] = (np.random.random(cf.controls) * (cf.CVrange[:,1] - cf.CVrange[:,0]) + cf.CVrange[:,0]) * 1000 #ivvi uses mV


if cf.device == 'chip':
    # Initialize instruments and set first random controls to avoid transients in main acquisition script
    ivvi = IVVIrack.initInstrument()
    IVVIrack.setControlVoltages(ivvi, theta[0, :])
elif cf.device == 'NN':  
    # If NN is used as proxy, load network
    net = lightNNet(cf.main_dir + cf.NN_name)
    
    
# Initialize save directory
saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# Initialize main figure
mainFig = PlotBuilder.initMainFigEvolution(cf.controls, cf.n, cf.CVlabels, [cf.CVrange[:,1]]*cf.controls)

# Main acquisition loop
for k in range(0, cf.n): 
    # Initialize parameters and control voltages
    ak = cf.a / (k + 1 + cf.A) ** cf.alpha
    ck = cf.c / (k + 1) ** cf.gamma
    for i in range(cf.controls):
        delta[k, i] = 2 * round(random.uniform(0, 1)) - 1
    thetaplus[k, :] = theta[k,:] + ck * delta[k, :]
    thetaminus[k, :] = theta[k,:] - ck * delta[k, :]
    
    # Check constraints on the range of theta
    for i in range(cf.controls):    
        thetaplus[k, i] = min(cf.CVrange[i,1]*1000, thetaplus[k, i])
        thetaminus[k, i] = max(cf.CVrange[i,0]*1000, thetaminus[k, i])
    
    # Measure for both CVs
    x_scaled = x * cf.inputScaling + cf.inputOffset #TODO: change to optimizable parameter
       
    
    if cf.device == 'chip':
        IVVIrack.setControlVoltages(ivvi, thetaplus[k, :])
        time.sleep(0.3)
        outputplus[k, :] = cf.gainFactor * InstrumentImporter.nidaqIO.IO(x_scaled, cf.fs)  
        time.sleep(0.1)
        IVVIrack.setControlVoltages(ivvi, thetaminus[k, :])
        time.sleep(0.1)
        outputminus[k, :] = cf.gainFactor * InstrumentImporter.nidaqIO.IO(x_scaled, cf.fs)
    elif cf.device == 'NN':
        ##############################
        # also set CVs
        inputsPlus = thetaplus[k][:,np.newaxis] * np.ones(x_scaled.shape[1])/1000
        inputsMinus = thetaminus[k][:,np.newaxis] * np.ones(x_scaled.shape[1])/1000
        for j in range(len(cf.inputIndex)):
            inputsPlus = np.insert(inputsPlus, cf.inputIndex[j], x_scaled[j,:], axis=0)
            inputsMinus = np.insert(inputsMinus, cf.inputIndex[j], x_scaled[j,:], axis=0)
        
        outputplus[k,:] = net.outputs(torch.from_numpy(inputsPlus.T).to(torch.float)) * net.info['conversion']
        outputminus[k,:] = net.outputs(torch.from_numpy(inputsMinus.T).to(torch.float)) * net.info['conversion']
        

        
    # Calculate loss function
    yplus[k] = cf.loss(outputplus[k, :], target, w)
    yminus[k] = cf.loss(outputminus[k, :], target, w)
    
    # Plot current controls (thetaminus)
    PlotBuilder.currentGenomeEvolution(mainFig, thetaminus)
    
    # Plot progress output (on minus)
    PlotBuilder.currentOutputEvolution(mainFig,
                                               t,
                                               target,
                                               outputminus[k, :],
                                               k + 1, 1,
                                               yminus[k]) # Loss is used here instead of fitness, so lower is better
    
    # Plot the loss of this iteration (of yminus)
    PlotBuilder.lossMainSPSA(mainFig, yminus)
    # Calculate gradient and update CV
    ghat[k, :] = (yplus[k] - yminus[k]) / (2 * ck * delta[k, :])
    theta[k + 1,:] = theta[k, :] - ak * ghat[k, :]

    # If the device clips, yplus = yminus so ghat = 0. We then want to randomly re-initialize one of the controls
    # and hope that the device won't clip anymore
    if ghat[k, 0] == 0.:
        ctrl = np.random.randint(cf.controls)
        theta[k + 1, ctrl] = random.uniform(cf.CVrange[ctrl,0], cf.CVrange[ctrl,1])

    # Check constraints on the range of theta
    for i in range(cf.controls):    
        theta[k+1,i] = min(cf.CVrange[i,1]*1000, theta[k+1,i])
        theta[k+1,i] = max(cf.CVrange[i,0]*1000, theta[k+1,i])
    
SaveLib.saveExperiment(cf.configSrc, saveDirectory,
                     controls = theta,
                     controlsPlus = thetaplus,
                     controlsMinus = thetaminus,
                     outputminus = outputminus,
                     outputplus = outputplus,
                     yminus = yminus,
                     yplus = yplus,
                     t = t,
                     x = x,
                     target = target,
                     ghat = ghat,
                     delta = delta)

if cf.device == 'chip':
    InstrumentImporter.reset(0,0)
