from instruments.niDAQ import nidaqIO
from instruments.ADwin import adwinIO
import modules.Evolution as Evolution
from instruments.DAC import IVVIrack
import modules.PlotBuilder as PlotBuilder
import time
import matplotlib.pyplot as plt

# temporary imports
import numpy as np

# define input
t = 1
Fs = 200000
frequency = 1
time_array = np.linspace(0, t, Fs*t)
inp = np.sin(2*np.pi*frequency*time_array)

adw = adwinIO.initInstrument()

#x = nidaqIO.IO(inp, Fs)
x = adwinIO.IO(adw, inp, Fs)

plt.plot(time_array, inp)


plt.figure()

plt.plot(time_array, x)
plt.show()