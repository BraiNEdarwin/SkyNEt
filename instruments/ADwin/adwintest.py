import SkyNEt.modules.SaveLib as SaveLib
import matplotlib.pyplot as plt
from SkyNEt.instruments import InstrumentImporter
import numpy as np
import os
import matplotlib.pyplot as plt

f = 1
Fs = 10000
t = np.linspace(0, 10, 10*Fs)
inputSignal = np.sin(2*np.pi * f * t)

adwin = InstrumentImporter.adwinIO.initInstrument()
Output = InstrumentImporter.adwinIO.IO(adwin, inputSignal, Fs)


# Plot the IV curve.
plt.figure()
plt.plot(t[:len(Output)], Output)
plt.show()

# Final reset
InstrumentImporter.reset(0, 0)