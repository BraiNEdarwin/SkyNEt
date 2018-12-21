import SkyNEt.modules.SaveLib as SaveLib
import matplotlib.pyplot as plt
from SkyNEt.instruments import InstrumentImporter
import SkyNEt.modules.SaveLib
import numpy as np
import os
import matplotlib.pyplot as plt

filepath = r'D:\data\Mark\NoiseSystem\\'
name = 'NoiseFloor_A1=100M_A2=1_R=1M_float'
#name = 'ADwin_test_I1_O2'

plotFlag = 1
saveFlag = 1

if saveFlag:
	saveDirectory = SaveLib.createSaveDirectory(filepath, name)
	configSrc = os.path.dirname(os.path.abspath(__file__))


f = 1
Fs = 10000
t = np.linspace(0, 10, 10*Fs)
#inputSignal = np.zeros((1,10*Fs))
#inputSignal[0] = np.sin(2*np.pi * f * t)

inputSignal = -3*np.ones((1, 1000))
inputSignal2 = -2*np.ones((1, 1000))
inputSignal3 = -1*np.ones((1, 1000))
inputSignal4 = 0*np.ones((1, 1000))
inputSignal5 = 1*np.ones((1, 1000))
inputSignal6 = 2*np.ones((1, 1000))
inputSignal7 = 3*np.ones((1, 1000))

Input = np.concatenate((inputSignal, inputSignal2, inputSignal3, inputSignal4, inputSignal5, inputSignal6, inputSignal7), axis = 1)
Input = np.zeros((1,10*Fs))

adwin = InstrumentImporter.adwinIO.initInstrument()

Output = InstrumentImporter.adwinIO.IO(adwin, Input, Fs)

InstrumentImporter.adwinIO.reset(adwin)

if plotFlag:
	# Plot the IV curve.
	plt.figure()
	plt.plot(Input[0])
	plt.plot(Output[0,:])
	plt.show()

if saveFlag:
	SaveLib.saveExperiment(configSrc, saveDirectory, input = Input, output = Output)


