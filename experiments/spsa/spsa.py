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
theta = np.zeros(cf.controls)   # Control voltages
delta = np.zeros(cf.controls)   # Direction of optimization step
for i in range(cf.controls):
    theta[i] = random.uniform(cf.CVrange[0], cf.CVrange[1])

# Initialize save directory
saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# Initialize main figure
mainFig = PlotBuilder.initMainFigEvolution(cf.controls, cf.n, cf.CVlabels, [cf.CVrange]*cf.controls)

# Initialize instruments and set first random controls to avoid transients in main acquisition script
ivvi = IVVIrack.initInstrument()
IVVIrack.setControlVoltages(ivvi, theta)

# Main acquisition loop
for k in range(0, cf.n): 
    # Initialize parameters and control voltages
    ak = cf.a / (k + 1 + cf.A) ** cf.alpha
    ck = cf.c / (k + 1) ** cf.gamma
    for i in range(cf.controls):
        delta[i] = 2 * round(random.uniform(0, 1)) - 1
    thetaplus = theta + ck * delta
    thetaminus = theta - ck * delta
    
    # Check constraints on the range of theta
    for i in range(cf.controls):    
        thetaplus[i] = min(cf.CVrange[1], thetaplus[i])
        thetaminus[i] = max(cf.CVrange[0], thetaminus[i])
    
    # Measure for both CVs
    x_scaled = 2 * (x - 0.5) * cf.CVrange[1]/1000 #TODO: change to optimizable parameter
    
    
    IVVIrack.setControlVoltages(ivvi, thetaplus)
    time.sleep(0.3)
    outputplus = cf.gainFactor * InstrumentImporter.nidaqIO.IO(x_scaled, cf.fs) 
    time.sleep(0.1)
    IVVIrack.setControlVoltages(ivvi, thetaminus)
    time.sleep(0.1)
    outputminus = cf.gainFactor * InstrumentImporter.nidaqIO.IO(x_scaled, cf.fs)
    
    # Calculate loss function
    yplus[k] = cf.loss(outputplus, target)
    yminus[k] = cf.loss(outputminus, target)
    
    # Calculate gradient and update CV
    ghat = (yplus[k] - yminus[k]) / (2 * ck * delta)
    theta = theta - ak * ghat
    # Check constraints on the range of theta
    for i in range(cf.controls):    
        theta[i] = min(cf.CVrange[1], theta[i])
        theta[i] = max(cf.CVrange[0], theta[i])
    
    # Plot progress (on yminus)
    PlotBuilder.currentOutputEvolution(mainFig,
                                               t,
                                               target,
                                               outputminus[0],
                                               1, k + 1,
                                               1/yminus[k])
    
   # PlotBuilder.updateMainFigEvolution(mainFig,
    #                                   theta,
     #                                  outputminus[0],
      #                                 outputminus[0],
       #                                k + 1,
        #                               t,
         #                              target,
          #                             outputminus[0],
           #                            w)
    
SaveLib.saveExperiment(cf.configSrc, saveDirectory,
                     CV = theta,
                     outputminus = outputminus,
                     lossminus = yminus,
                     lossplus = yplus,
                     t = t,
                     x = x,
                     target = target)

InstrumentImporter.reset(0,0)
