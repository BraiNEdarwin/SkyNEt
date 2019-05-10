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

N = 100

output = np.zeros(N)

# Set compliances
keithley.compliancei.set(1E-6)
keithley.compliancev.set(4)

# Turn keithley output on
keithley.output.set(1)

start = time.time()
for ii in range(len(output)):
    # Record current
#    time.sleep(0.05)
    output[ii] = keithley.curr()
    print('got currernt %i' % ii)

print( time.time()-start)
# Turn keithley output off
keithley.output.set(0)
keithley.close()

plt.figure()
plt.plot(np.arange(N), output*1e9)
plt.ylabel('output nA')
plt.xlabel('sample')