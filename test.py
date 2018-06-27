# from instruments.DAC import IVVIrack
import time
# temporary imports
import numpy as np
from instruments.ADwin import adwinIO
import matplotlib.pyplot as plt
# from instruments.DAC import IVVIrack
# from instruments.Keithley2000 import Keithley_2000

# ivvi = IVVIrack.initInstrument()
# Keithley_2000.__init__(elkeef, "GPIB:17")
adwin = adwinIO.initInstrument()

x = np.linspace(0, 1, 2000)
Fs = 1000

output = adwinIO.IO(adwin, x, Fs)

print(len(output))
# inputvoltages = np.array([500])
# IVVIrack.setinputVoltages(ivvi, inputVoltages)
# time.sleep(1)

# measureddata = Keithley_2000._read_next_value()
# output = measureddata[0]

# inputvoltages = np.array([1000])

# IVVIrack.setinputVoltages(ivvi, inputVoltages)
# time.sleep(1)

# measureddata = Keithley_2000._read_next_value()
# output = measureddata[1]


plt.figure()
plt.plot(output)
plt.show()
