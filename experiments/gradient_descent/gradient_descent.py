# -*- coding: utf-8 -*-

"""
Created on Sun Jan 27 12:29:46 2019

This script applies gradient descent directly on either the device or a simulation of the device.
For now it is only possible to find Boolean logic with this experiment.

@author: Mark Boon
"""
import SkyNEt.experiments.gradient_descent.config_gradient_descent as config
import SkyNEt.modules.SaveLib as SaveLib
import SkyNEt.modules.PlotBuilder as PlotBuilder
from SkyNEt.modules.Nets.staNNet import staNNet

import numpy as np

# Initialize config object
cf = config.experiment_config()

if cf.device == 'chip':
    from SkyNEt.instruments import InstrumentImporter
    # Initialize save directory
    saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)
elif cf.device == 'NN':
    import torch


# Initialize input and target
#TOOD: create general problem loader and specify problem in config file
t = cf.InputGen[0]  # Time array
x = cf.InputGen[1]  # Array with input signals [inputs x time signal]
w = cf.InputGen[2]  # Weight array

target = cf.gainFactor * cf.InputGen[3]  # Target signal

# Initialize arrays
controls = np.zeros((cf.n + 1, cf.controls)) # array that keeps track of the controls used per iteration
controls[0,:] = np.random.random(cf.controls) * (cf.CVrange[:,1] - cf.CVrange[:,0]) + cf.CVrange[:,0] # First (random) controls 
m = np.zeros((cf.n+1,cf.controls))
v = np.zeros((cf.n+1,cf.controls))

inputs = np.zeros((cf.n + 1, cf.controls + cf.inputs, x.shape[1] + int(2 * cf.fs * cf.rampT)))
data = np.zeros((cf.n + 1, x.shape[1])) # '+1' bacause at the end of iterating the solution is measured without sine waves on top of the DC signal
IVgrad = np.zeros((cf.n + 1, cf.inputCases, cf.controls)) # dI_out/dV_in
EIgrad = np.zeros((cf.n + 1, int(np.sum(w)))) # gradient using cost function and the output (dE/dI)
EVgrad = np.zeros((cf.n + 1, cf.controls)) # gradient from cost function to control voltage (dE/dV)
error = np.ones(cf.n + 1)
x_scaled = x * cf.inputScaling + cf.inputOffset # Note that in the current script the inputs cannot be controlled by the algorithm but are constant

# If NN is used as proxy, load network
if cf.device == 'NN':  
    net = staNNet(cf.main_dir + cf.NN_name)

# Initialize main figure
mainFig = PlotBuilder.initMainFigEvolution(cf.controls, cf.n, cf.controlLabels, cf.controls * [cf.CVrange[:,1]])

# Select indices that are used as controls
indices = []
for k in range(cf.controls + cf.inputs):
    if k not in cf.inputIndex: indices += [k]
    
# Main aqcuisition loop
for i in range(cf.n + 1):  
    # Apply DC control voltages:
    inputs[i, indices, int(cf.fs*cf.rampT):-int(cf.fs*cf.rampT)] = controls[i,:][:,np.newaxis] * np.ones(x.shape[1])    
    # For all except last iteration add sine waves on top of DC voltages
    if i != cf.n:
        inputs[i, indices, int(cf.fs*cf.rampT):-int(cf.fs*cf.rampT)] = inputs[i, indices, int(cf.fs*cf.rampT):-int(cf.fs*cf.rampT)] + np.sin(2 * np.pi * cf.freq[:,np.newaxis] * t) * cf.A_in[:,np.newaxis]
    
    # Add (boolean) input at the correct index of the input matrix:
    for j in range(len(cf.inputIndex)):
        inputs[i,cf.inputIndex[j],:] = np.concatenate((np.zeros(int(cf.fs*cf.rampT)), x_scaled[j,:], np.zeros(int(cf.fs*cf.rampT))))
    
    # Add ramping up and ramping down the voltages at start and end of iteration (if measuring on real device)
    if cf.device == 'chip':
        for j in range(inputs.shape[1]):
            inputs[i, j, 0:int(cf.fs*cf.rampT)] = np.linspace(0, inputs[i, j, int(cf.fs*cf.rampT)], int(cf.fs*cf.rampT))
            inputs[i, j, -int(cf.fs*cf.rampT):] = np.linspace(inputs[i, j, -int(cf.fs*cf.rampT + 1)], 0, int(cf.fs*cf.rampT))    
        
    # Measure output
    if cf.device == 'chip':
        dataRamped = InstrumentImporter.nidaqIO.IO_cDAQ(inputs[i,:,:], cf.fs) * cf.gainFactor
        data[i,:] = dataRamped[0, int(cf.fs*cf.rampT):-int(cf.fs*cf.rampT)]   # Cut off the ramping up and down part
    elif cf.device == 'NN':
        data[i,:] = net.outputs(torch.from_numpy(inputs[i,:,int(cf.fs*cf.rampT):-int(cf.fs*cf.rampT)].T).to(torch.float))

    # Calculate dE/dI   
    EIgrad[i,:] = cf.gradFunct(data[i], target, w) #TODO: create minibatch GD
    
    # Split the input cases into different samples for determining gradients
    data_split = np.zeros((cf.inputCases, int(cf.fs*cf.signallength/cf.inputCases)))
    target_split = np.zeros((cf.inputCases, int(cf.fs*cf.signallength/cf.inputCases)))
    sign = np.zeros((cf.inputCases, controls.shape[1]))
    
    # Lock-in technique to determine gradients
    ##x_ref = np.arange(0.0, cf.signallength, 1/cf.fs)
    for k in range(cf.inputCases):
        data_split[k] = data[i, round(k*cf.fs*(cf.edgelength + cf.signallength/cf.inputCases)) : round(cf.fs*(k*cf.edgelength + (k+1)*cf.signallength/cf.inputCases))]
        target_split[k] = target[round(k*cf.fs*(cf.edgelength + cf.signallength/cf.inputCases)) : round(cf.fs*(k*cf.edgelength + (k+1)*cf.signallength/cf.inputCases))]
          
        IVgrad[i,k,:] = cf.lock_in_gradient(data_split[k], cf.freq, cf.A_in)
        EVgrad[i] += np.mean(EIgrad[i, int(k*cf.signallength*cf.fs//cf.inputCases):int((k+1)*cf.signallength*cf.fs//cf.inputCases)][:,np.newaxis] * IVgrad[i,k,:], axis=0)
    
    
    if i < cf.n-1:
        controls[i+1,:], m[i+1], v[i+1] = cf.optimizer(i, controls[i,:], EVgrad[i], cf.eta, m[i], v[i], cf.beta_1, cf.beta_2)
        # Make sure that the controls stay within the specified range
        for j in range(controls.shape[1]):
            controls[i+1, j] = min(cf.CVrange[j, 1], controls[i+1, j])
            controls[i+1, j] = max(cf.CVrange[j, 0], controls[i+1, j])
    elif i == cf.n-1:
        controls[i+1,:] = controls[i, :] # Keep same controls, last measure is last iteration but without sine waves

    # If output is clipped, reinitializee:
    if abs(np.mean(data[i,:])) > 3.5 * cf.amplification/cf.postgain:
        controls[i+1,:] = np.random.random(cf.controls) * (cf.CVrange[:,1] - cf.CVrange[:,0]) + cf.CVrange[:,0]
    

    # Plot output, error, controls
    error[i] = cf.errorFunct(data[i,:], target, w)     
    PlotBuilder.currentGenomeEvolution(mainFig, controls[i,:])
    PlotBuilder.currentOutputEvolution(mainFig,
                                               t,
                                               target,
                                               data[i,:],
                                               1, i + 1,
                                               error[i])
    PlotBuilder.fitnessMainEvolution(mainFig,
                                             error[:,np.newaxis],
                                             i + 1)

if cf.device == 'chip':
    SaveLib.saveExperiment(cf.configSrc, saveDirectory,
                           controls = controls,
                           inputs = inputs,
                           output = data,
                           t = t,
                           w = w,
                           target = target,
                           fs = cf.fs,
                           inputIndex = cf.inputIndex,
                           signallength = cf.signallength,
                           eta = cf.eta,
                           freq = cf.freq,
                           n = cf.n,
                           A_in = cf.A_in,
                           x_scaled = x_scaled,
                           error = error,
                           IVgrad = IVgrad,
                           EVgrad = EVgrad,
                           EIgrad = EIgrad)
    
PlotBuilder.finalMain(mainFig)

if cf.device == 'chip':
    InstrumentImporter.reset(0, 0)  