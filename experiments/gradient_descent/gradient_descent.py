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
elif cf.device == 'NN':
    import torch

# Initialize save directory
saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# Initialize input and target
t = cf.InputGen()[0]  # Time array
x = np.asarray(cf.InputGen()[1:3])  # Array with P and Q signal
w = cf.InputGen()[3]  # Weight array
target = cf.gainFactor * cf.targetGen()[1]  # Target signal

# Initialize arrays
controls = np.zeros((cf.n + 1, cf.controls)) # array that keeps track of the controls used per iteration
controls[0,:] = np.random.random(cf.controls) * (cf.CVrange[:,1] - cf.CVrange[:,0]) + cf.CVrange[:,0] # First (random) controls
inputs = np.zeros((cf.n + 1, cf.controls + cf.inputs, x.shape[1] + int(2 * cf.fs * cf.rampT)))
data = np.zeros((cf.n + 1, x.shape[1])) # '+1' bacause at the end of iterating the solution is measured without sine waves on top of the DC signal
phases = np.zeros((cf.n + 1, cf.inputCases, cf.controls)) # iterations x # input cases x # controls 
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
    
    # Apply the sine waves on top of the control voltages:
    inputs[i, indices, int(cf.fs*cf.rampT):-int(cf.fs*cf.rampT)] = controls[i,:][:,np.newaxis] * np.ones(x.shape[1]) 
    
    # For all except last iteration add sine waves on top of DC voltages
    if i != cf.n:
        inputs[i, indices, int(cf.fs*cf.rampT):-int(cf.fs*cf.rampT)] = inputs[i, indices, int(cf.fs*cf.rampT):-int(cf.fs*cf.rampT)] + np.sin(2 * np.pi * cf.freq[:,np.newaxis] * t) * cf.waveAmplitude
    
    # Add (boolean) input at the correct index of the input matrix:
    for j in range(len(cf.inputIndex)):
        inputs[i,cf.inputIndex[j],:] = np.concatenate((np.zeros(int(cf.fs*cf.rampT)), x_scaled[j,:], np.zeros(int(cf.fs*cf.rampT))))
    
    # Add ramping up and ramping down the voltages at start and end of iteration (if measuring on real device)
    if cf.device == 'chip':
        for j in range(inputs.shape[1]):
            inputs[i, j, 0:int(cf.fs*cf.rampT)] = np.linspace(0, inputs[i, j, int(cf.fs*cf.rampT)], int(cf.fs*cf.rampT))
            inputs[i, j, -int(cf.fs*cf.rampT):] = np.linspace(inputs[i, j, -int(cf.fs*cf.rampT)], 0, int(cf.fs*cf.rampT))    
        
    # Measure output
    if cf.device == 'chip':
        dataRamped = InstrumentImporter.nidaqIO.IO_cDAQ(inputs[i,:,:], cf.fs) * cf.gainFactor
        data[i,:] = dataRamped[0, int(cf.fs*cf.rampT):-int(cf.fs*cf.rampT)]   # Cut off the ramping up and down part
    elif cf.device == 'NN':
        data[i,:] = net.outputs(torch.from_numpy(inputs[i,:,int(cf.fs*cf.rampT):-int(cf.fs*cf.rampT)].T).to(torch.float))

    # Calculate dE/dI   
    EIgrad[i,:] = cf.gradFunct(data[i], target, w) 
    
    # Split the input cases into different samples for determining gradients
    data_split = np.zeros((cf.inputCases, int(cf.fs*cf.signallength/cf.inputCases)))
    target_split = np.zeros((cf.inputCases, int(cf.fs*cf.signallength/cf.inputCases)))
    sign = np.zeros((cf.inputCases, controls.shape[1]))
    
    # Lock-in technique to determine gradients
    x_ref = np.arange(0.0, cf.signallength, 1/cf.fs)
    #TODO: This is wrong! the input wave isn't 'reset' for each case, so phases are wrong
    # Solve this by either letting x_ref contain all cases, or by defining the input waves on the control
    # as phase=0 at every start of an input case.
    
    for k in range(cf.inputCases):
        data_split[k] = data[i, round(k*cf.fs*(cf.edgelength + cf.signallength/cf.inputCases)) : round(cf.fs*(k*cf.edgelength + (k+1)*cf.signallength/cf.inputCases))]
        target_split[k] = target[round(k*cf.fs*(cf.edgelength + cf.signallength/cf.inputCases)) : round(cf.fs*(k*cf.edgelength + (k+1)*cf.signallength/cf.inputCases))]
                       
        for j in range(controls.shape[1]):
            y_ref1 = np.sin(cf.freq[j] * 2.0*np.pi*x_ref[k*int(cf.fs*cf.signallength/cf.inputCases) : (k+1)*int(cf.fs*cf.signallength/cf.inputCases)])          # Reference signal 1 (phase = 0)
            y_ref2 = np.sin(cf.freq[j] * 2.0*np.pi*x_ref[k*int(cf.fs*cf.signallength/cf.inputCases) : (k+1)*int(cf.fs*cf.signallength/cf.inputCases)] + np.pi/2) # Reference signal 2 (phase = pi/2)
        
            y1_out = y_ref1*(data_split[k] - np.mean(data_split[k]))
            y2_out = y_ref2*(data_split[k] - np.mean(data_split[k]))
            
            amp1 = np.mean(y1_out)
            amp2 = np.mean(y2_out)
            IVgrad[i,k,j] = 2*np.sqrt(amp1**2 + amp2**2) / cf.waveAmplitude # Amplitude of output wave (nA) / amplitude of input wave (V)
            phases[i,k,j] = np.arctan2(amp2,amp1)*180/np.pi
            sign[k,j] = 1 if abs(phases[i, k, j]) < cf.phase_thres else -1
        
        ## Fourier transform on output data to find dI/dV
        #yf = scipy.fftpack.fft(data_split[k,:], cf.fft_N)
        #IVgrad[i,k,:] = 2.0 / cf.fft_N * np.abs(yf[(cf.freq*cf.fft_N/cf.fs).astype(int)]) / cf.waveAmplitude # Amplitude of output wave / amplitude of input wave
        #phases[i,k,:] = np.arctan2(yf[(cf.freq*cf.fft_N/cf.fs).astype(int)].imag, yf[(cf.freq*cf.fft_N/cf.fs).astype(int)].real) + np.pi/2 # Add pi/2 since sine waves are used
    
        #for j in range(controls.shape[1]):
        #    sign[k,j] = -1 if abs(phases[i, k, j] - np.pi) < cf.phase_thres else 1
        
        
        # Multiply dE/dI and dI/dV to obtain the gradient w.r.t. control voltages
        EVgrad[i] += np.mean(EIgrad[i, k*cf.signallength*cf.fs//cf.inputCases:(k+1)*cf.signallength*cf.fs//cf.inputCases][:,np.newaxis] * (sign[k,:] * IVgrad[i,k,:]), axis=0)
            
    if i < cf.n-1:
        # Update scheme
        controls[i+1,:] = controls[i, :] - cf.eta * EVgrad[i]
        
        # Make sure that the controls stay within the specified range
        for j in range(controls.shape[1]):
            controls[i+1, j] = min(cf.CVrange[j, 1], controls[i+1, j])
            controls[i+1, j] = max(cf.CVrange[j, 0], controls[i+1, j])
    elif i == cf.n-1:
        controls[i+1,:] = controls[i, :] # Keep same controls, last measure is last iteration but without sine waves
    
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


SaveLib.saveExperiment(cf.configSrc, saveDirectory,
                       controls = controls,
                       output = data,
                       t = t,
                       x_scaled = x_scaled,
                       error = error,
                       IVgrad = IVgrad,
                       EVgrad = EVgrad,
                       EIgrad = EIgrad)
    
PlotBuilder.finalMain(mainFig)

if cf.device == 'chip':
    InstrumentImporter.reset(0, 0)  