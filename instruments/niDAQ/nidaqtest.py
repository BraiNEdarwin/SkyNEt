import SkyNEt.modules.SaveLib as SaveLib
import matplotlib.pyplot as plt
from SkyNEt.instruments import InstrumentImporter
import numpy as np
import os

# Specify inputs
fs = 1000
#Input = np.zeros((1,900))
#for i in range(0,900):
#	Input[0,i] = i*0.001

#Input[0,0] = 1
#Input[0,-1] = 0
sampleTime = 100
freq = 20
t = np.linspace(0,sampleTime,fs*sampleTime)


sampleTime = 100
freq = 20
t = np.linspace(0, sampleTime, fs*sampleTime)

Input = np.zeros((1,fs*sampleTime))
Input[0,:] = np.sin(2*np.pi*freq*t)
Input[0,0] = 2
Input[0,-1] = 2

# Measure using the device specified in the config class.
Output = InstrumentImporter.nidaqIO.IO_cDAQ(Input, fs)

# Plot the IV curve.
plt.figure()
plt.plot(Output[0])
plt.plot(Input[0])
plt.show()

# Final reset
InstrumentImporter.reset(0, 0)
