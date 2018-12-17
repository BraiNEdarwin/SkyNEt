# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 08:29:46 2018

@author: Mark Boon
"""

import SkyNEt.experiments.spsa.config_spsa as config
import SkyNEt.modules.SaveLib as SaveLib
from SkyNEt.instruments.niDAQ import nidaqIO
from SkyNEt.instruments.DAC import IVVIrack
import SkyNEt.modules.PlotBuilder as PlotBuilder

import time
import numpy as np
import random
import signal
import sys

def reset(signum, frame):
    '''
    This functions performs the following reset tasks:
    - Set IVVI rack DACs to zero
    - Apply zero signal to the NI daq
    - Apply zero signal to the ADwin
    '''
    try:
        global ivvi
        ivvi.set_dacs_zero()
        print('ivvi DACs set to zero')
        del ivvi  # Test if this works!
    except:
        print('ivvi was not initialized, so also not reset')
			
    try:
        nidaqIO.reset_device()
        print('nidaq has been reset')
    except:
        print('nidaq not connected to PC, so also not reset')

    try:
        global adw
        reset_signal = np.zeros((2, 40003))
        adwinIO.IO_2D(adw, reset_signal, 1000)
    except:
        print('adwin was not initialized, so also not reset')

    # Finally stop the script execution
    sys.exit()
        
        
#%% Initialization
signal.signal(signal.SIGINT, reset)

# Initialize config object
cf = config.experiment_config()

# Initialize input and target
t = cf.InputGen()[0]  # Time array
x = np.asarray(cf.InputGen()[1:3])  # Array with P and Q signal
w = cf.InputGen()[3]  # Weight array
target = cf.TargetGen()[1]  # Target signal

# Initialize at arrays
yplus = np.zeros(cf.n)
yminus = np.zdros(cf.n)
theta = np.zeros(cf.controls)   # Control voltages
delta = np.zeros(cf.controls)   # Direction of optimization step
for i in range(cf.controls):
    theta[i] = random.uniform(cf.CVrange[0], cf.CVrange[1])

# Initialize save directory
saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# Initialize main figure
mainFig = PlotBuilder.initMainFigEvolution(cf.controls, cf.n, cf.CVlabels, [cf.CVrange]*cf.controls)

# Initialize instruments
ivvi = IVVIrack.initInstrument()


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
    x_scaled = 2 * (x - 0.5) * cf.range[1] #TODO: change to optimizable parameter
    
    IVVIrack.setControlVoltages(ivvi, thetaplus)
    time.sleep(0.3)
    outputplus = nidaqIO.IO_2D(x_scaled, cf.fs) #TODO: Check for adwin and fs
    time.sleep(0.1)
    IVVIrack.setControlVoltages(ivvi, thetaminus)
    time.sleep(0.3)
    outputminus = nidaqIO.IO_2D(x_scaled, cf.fs)
    
    # Calculate loss function
    yplus[k] = cf.loss(outputplus)
    yminus[k] = cf.loss(outputminus)
    
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
                                               outputminus,
                                               1, k + 1,
                                               1/yminus[k])
    
    PlotBuilder.updateMainFigEvolution(mainFig,
                                       theta,
                                       outputminus,
                                       outputminus,
                                       k + 1,
                                       t,
                                       cf.amplification*target,
                                       outputminus,
                                       w)
    
    
SaveLib.saveExperiment(cf.configSrc, saveDirectory,
                     CV = theta,
                     outputminus = outputminus,
                     lossminus = yminus,
                     lossplus = yplus,
                     t = t,
                     x = x,
                     amplified_target = cf.amplification*target)

reset(0,0)
