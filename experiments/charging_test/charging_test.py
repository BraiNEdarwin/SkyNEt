# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:51:29 2018
Script to test whether we can charge up the floating substrate to change the output behavior.
Written such that it can be used for the cDAQ, not compatible with other meaasure devices.

@author: Mark
"""
# Import packages
import SkyNEt.modules.SaveLib as SaveLib
from SkyNEt.instruments import InstrumentImporter
import time
import SkyNEt.experiments.charging_test.config_charging_test as config
# temporary imports
import numpy as np

#%% Initialization of saving config file
configSrc = config.__file__

# Initialize config object
cf = config.experiment_config()

# initialize save directory
saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# Initialize data
chargeOutputs = np.zeros((cf.amplitudes.shape[0], cf.fs* cf.chargeTime + cf.fs//2))
refOutputs = np.zeros((cf.amplitudes.shape[0]+1, cf.fs* cf.sampleTime + cf.fs//2))
staticOutputs = np.zeros((cf.amplitudes.shape[0]+1, cf.t_static.shape[0] + cf.fs))
rampDown = np.zeros((7,cf.fs//2)) 
rampUp = np.zeros((7,cf.fs//2)) 


t_charge = np.linspace(0, cf.fs*cf.chargeTime-1, cf.fs*cf.chargeTime)
t_sample = np.linspace(0, cf.fs*cf.sampleTime-1, cf.fs*cf.sampleTime)

'''
inputs = cf.generateSineWave(cf.freq, t_sample, cf.refAmplitude*np.ones(7), cf.fs, np.zeros(7))

for j in range(inputs.shape[0]):
	rampDown[j] = np.linspace(inputs[j,-1],0,cf.fs//2) # 0.5s ramp down after using waves
inputs = np.append(inputs, rampDown, axis=1)
print('Measuring first reference signal')
refOutputs[0] = InstrumentImporter.nidaqIO.IO_cDAQ(inputs, cf.fs)
'''
# Static measurement
inputs = cf.generateSineWave(cf.freq, cf.t_static, cf.refAmplitude*np.ones(7), cf.fs, np.zeros(7)) 
for j in range(inputs.shape[0]):
	rampUp[j] = np.linspace(0, inputs[j,0], cf.fs//2)
	rampDown[j] = np.linspace(inputs[j,-1], 0, cf.fs//2)
inputs = np.concatenate((rampUp,inputs,rampDown),axis=1)
print('Measuring first static signal')
staticOutputs[0] = InstrumentImporter.nidaqIO.IO_cDAQ(inputs, cf.fs)

for i in range(cf.amplitudes.shape[0]):
	# Charge measurment
	inputs = cf.generateSineWave(cf.freq, t_charge, cf.amplitudes[i]*np.ones(7), cf.fs, np.zeros(7))
	for j in range(inputs.shape[0]):
		rampDown[j] = np.linspace(inputs[j,-1],0,cf.fs//2) # 0.5s ramp down after using waves
	inputs = np.append(inputs, rampDown, axis=1)

	print('Charging with amplitude ' + str(cf.amplitudes[i]))
	chargeOutputs[i] = InstrumentImporter.nidaqIO.IO_cDAQ(inputs, cf.fs)

	'''
	# Reference measurement
	inputs = cf.generateSineWave(cf.freq, t_sample, cf.refAmplitude*np.ones(7), cf.fs, np.zeros(7))
	for j in range(inputs.shape[0]):
		rampDown[j] = np.linspace(inputs[j,-1],0,cf.fs//2) # 0.5s ramp down after using waves
	inputs = np.append(inputs, rampDown, axis=1)
	print('Measuring reference signal')
	refOutputs[i+1] = InstrumentImporter.nidaqIO.IO_cDAQ(inputs, cf.fs)

	'''
	# Static measurement
	inputs = cf.generateSineWave(cf.freq, cf.t_static, cf.refAmplitude*np.ones(7), cf.fs, np.zeros(7)) 
	for j in range(inputs.shape[0]):
		rampUp[j] = np.linspace(0, inputs[j,0], cf.fs//2)
		rampDown[j] = np.linspace(inputs[j,-1], 0, cf.fs//2)
	inputs = np.concatenate((rampUp,inputs,rampDown),axis=1)
	print('Measuring static signal')
	staticOutputs[i+1] = InstrumentImporter.nidaqIO.IO_cDAQ(inputs, cf.fs)


	# Save after every iteration
	print('Saving...')
	SaveLib.saveExperiment(cf.configSrc, saveDirectory, 
	                                refOutputs = refOutputs*cf.amplification/cf.postgain,
	                                chargeOutputs = chargeOutputs*cf.amplification/cf.postgain,
	                                staticOutputs = staticOutputs*cf.amplification/cf.postgain,
	                                staticInputs = inputs,
	                                freq = cf.freq,
	                                chargeTime = cf.chargeTime,
	                                sampleTime = cf.sampleTime,
	                                amplitudes = cf.amplitudes,
	                                refAmplitude = cf.refAmplitude,
	                                fs = cf.fs,
	                                phase = cf.phase,
	                                amplification = cf.amplification,
	                                electrodeSetup = cf.electrodeSetup,
	                                gain_info = cf.gain_info,
	                                filename = 'data')
	print('Done.')

InstrumentImporter.reset(0,0)

