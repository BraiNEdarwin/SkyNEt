# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:32:48 2019

@author: ljknoll
"""

from SkyNEt.instruments.Keithley2400 import Keithley2400

keithley = Keithley2400.Keithley_2400('keithley', 'GPIB0::11')

import time
import numpy as np
import matplotlib.pyplot as plt

output = np.zeros(100)

# Set compliances
keithley.compliancei.set(1E-6)
keithley.compliancev.set(4)

# Turn keithley output on
keithley.output.set(1)

start = time.time()
for ii in range(len(output)):
    # Set voltage
#    keithley.volt.set(Input[ii])

    # Record current
#    time.sleep(0.05)
    output[ii] = keithley.curr()

print( time.time()-start)
# Turn keithley output off
keithley.output.set(0)

plt.figure()
plt.plot(output)