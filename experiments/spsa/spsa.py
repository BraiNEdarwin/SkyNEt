# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 08:29:46 2018

This script applies the SPSA algorithm on the device.

@author: Mark Boon
"""

import SkyNEt.experiments.spsa.config_spsa as config
import SkyNEt.modules.SaveLib as SaveLib
from SkyNEt.instruments import InstrumentImporter
from SkyNEt.instruments.DAC import IVVIrack
import SkyNEt.modules.PlotBuilder as PlotBuilder

import time
import numpy as np
import random

# Initialize config object
cf = config.experiment_config()

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
for i in range(cf.controls):
    theta[0, i] = random.uniform(cf.CVrange[0], cf.CVrange[1])

# Initialize save directory
saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# Initialize main figure
mainFig = PlotBuilder.initMainFigEvolution(cf.controls, cf.n, cf.CVlabels, [cf.CVrange]*cf.controls)

# Initialize instruments and set first random controls to avoid transients in main acquisition script
ivvi = IVVIrack.initInstrument()
IVVIrack.setControlVoltages(ivvi, theta[0, :])

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
        thetaplus[k, i] = min(cf.CVrange[1], thetaplus[k, i])
        thetaminus[k, i] = max(cf.CVrange[0], thetaminus[k, i])
    
    # Measure for both CVs
    x_scaled = 2 * (x - 0.5) * cf.CVrange[1]/1000 #TODO: change to optimizable parameter
    
    
    IVVIrack.setControlVoltages(ivvi, thetaplus[k, :])
    time.sleep(0.3)
    outputplus[k, :] = cf.gainFactor * InstrumentImporter.nidaqIO.IO(x_scaled, cf.fs) 
    time.sleep(0.1)
    IVVIrack.setControlVoltages(ivvi, thetaminus[k, :])
    time.sleep(0.1)
    outputminus[k, :] = cf.gainFactor * InstrumentImporter.nidaqIO.IO(x_scaled, cf.fs)
    
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
        ctrl =  np.random.randint(cf.controls)
        theta[k + 1, ctrl] = random.uniform(cf.CVrange[0], cf.CVrange[1])

    # Check constraints on the range of theta
    for i in range(cf.controls):    
        theta[k+1,i] = min(cf.CVrange[1], theta[k+1,i])
        theta[k+1,i] = max(cf.CVrange[0], theta[k+1,i])

    
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

InstrumentImporter.reset(0,0)
