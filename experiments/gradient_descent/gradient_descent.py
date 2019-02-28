# -*- coding: utf-8 -*-

"""
Created on Sun Jan 27 12:29:46 2019

This script applies gradient descent directly on the device.
For now it is only possible to find boolean logic with this experiment

@author: Mark Boon
"""
import SkyNEt.experiments.gradient_descent.config_gradient_descent as config
import SkyNEt.modules.SaveLib as SaveLib
from SkyNEt.instruments import InstrumentImporter
import SkyNEt.modules.PlotBuilder as PlotBuilder

import numpy as np
import scipy.fftpack

# Initialize config object
cf = config.experiment_config()

# Initialize save directory
saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# Initialize input and target
t = cf.InputGen()[0]  # Time array
x = np.asarray(cf.InputGen()[1:3])  # Array with P and Q signal
w = cf.InputGen()[3]  # Weight array
target = cf.gainFactor * cf.targetGen()[1]  # Target signal

# Initialize arrays
controls = np.zeros((cf.n, cf.controls)) # array that keeps track of the controls used per iteration
controls[0,:] = (np.random.random(cf.controls) - 0.5) * 2 * cf.Vrange[1] # First (random) controls
inputs = np.zeros((cf.controls, x.shape[1] + int(2 * cf.fs * cf.rampT)))
data = np.zeros((cf.n, x.shape[1]))
error = np.ones(cf.n)
x_scaled = x * cf.Vrange[1] # Note that in the current script the inputs cannot be controlled by the algorithm

# Initialize main figure
mainFig = PlotBuilder.initMainFigEvolution(cf.controls, cf.n, cf.controlLabels, cf.controls * [cf.Vrange])

# Main aqcuisition loop
for i in range(cf.n):
    # Apply the sine waves on top of the control voltages:
    inputs[:, int(cf.fs*cf.rampT):-int(cf.fs*cf.rampT)]  = np.dot(controls[:, i:i+1], np.ones((1, x.shape[1]))) + np.sin(2 * np.pi * np.dot(t[:,np.newaxis],cf.freq[np.newaxis,:])) * cf.waveAmplitude
    
    # Add (boolean) input at the correct index of the input matrix:
    for j in range(cf.inputIndex):
        inputs = np.insert(inputs, cf.inputIndex[j], np.concatenate((np.zeros(int(cf.fs*cf.rampT)), x_scaled[j,:], np.zeros(int(cf.fs*cf.rampT)))), axis=0)
    
    # Add ramping up and ramping down the voltages at start and end of iteration
    for j in range(inputs.shape[0]):
        inputs[j, 0:int(cf.fs*cf.rampT)] = np.linspace(0, inputs[j,int(cf.fs*cf.rampT)], int(cf.fs*cf.rampT))
        inputs[j, -int(cf.fs*cf.rampT):] = np.linspace(inputs[j,-int(cf.fs*cf.rampT)], 0, int(cf.fs*cf.rampT))    
        
    # Measure output
    dataRamped = InstrumentImporter.nidaqIO.IO_cDAQ(inputs, cf.fs) * cf.gainFactor
    data[i,:] = dataRamped[int(cf.fs*cf.rampT),-int(cf.fs*cf.rampT)]   # Cut off the ramping up and down part

    # Fourier transform on output data to find dI/dV
    yf = scipy.fftpack.fft(data[i,:], cf.fft_N)
    gradients = 2.0 / cf.fft_N * np.abs(yf[round(cf.freq*cf.fft_N/cf.fs)]) / cf.waveAmplitude # Amplitude of output wave / amplitude of input wave
    phases = np.arctan2(yf[round(cf.freq*cf.fft_N/cf.fs)].imag, yf[round(cf.freq*cf.fft_N/cf.fs)].real) + np.pi/2 # Add pi/2 since sine waves are used
    
    # Process the FFT and determine the controls for the next step
    sign = np.zeros(controls.shape[0])
    for j in range(controls.shape[0]):
        sign[j] = -1 if abs(phases[j] - np.pi) < cf.phase_thres else 1
    
    controls[:, i+1] = controls[:, i] - cf.eta * sign * cf.errorFunct(data[i,:], target, gradients, w)
    
    # Make sure that the controls stay within their range
    for j in range(controls.shape[0]):
        controls[j, i+1] = min(cf.CVrange[1], controls[j, i+1])
        controls[j, i+1] = max(cf.CVrange[0], controls[j, i+1])
    
    # Plot output, error, controls
    error[i] = np.sum((data[i,:] - t) * w)**2/np.sum(w)     
    PlotBuilder.currentGenomeEvolution(mainFig, controls[:,i])
    PlotBuilder.currentOutputEvolution(mainFig,
                                               t,
                                               target,
                                               data[i,:],
                                               1, i + 1,
                                               error[i])
    PlotBuilder.fitnessMainEvolution(mainFig,
                                             error[:,np.newaxis],
                                             i + 1)

SaveLib.saveExperiment(cf.configSrc, saveDirectory,
                       controls = controls,
                       output = data,
                       t = t,
                       x_scaled = x_scaled,
                       error = error)
    
PlotBuilder.finalMain(mainFig)

InstrumentImporter.reset(0, 0)  