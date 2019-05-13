# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:32:48 2019

@author: ljknoll
"""

from SkyNEt.instruments.Keithley2400 import Keithley2400
from SkyNEt.instruments import InstrumentImporter

keithley = Keithley2400.Keithley_2400('keithley', 'GPIB0::11')

import time
import numpy as np
import matplotlib.pyplot as plt

N = 20

ser = InstrumentImporter.switch_utils.init_serial('COM3')
InstrumentImporter.switch_utils.connect_single_device(ser, 5)
#print('INFO: Connected device %i' % )


input_data = np.zeros((N,7))
input_data[:,0] = np.linspace(-0.5,1,N)
output = np.zeros(N)
sdev = InstrumentImporter.nidaqIO.IO_cDAQ(nr_channels=7)


# Set compliances
keithley.compliancei.set(1E-6)
keithley.compliancev.set(4)

# Turn keithley output on
keithley.output.set(1)

start = time.time()
for ii,x in enumerate(input_data):
    sdev.ramp(x)
    # Record current
#    time.sleep(0.05)
    output[ii] = -keithley.curr()

print( time.time()-start)
# Turn keithley output off
keithley.output.set(0)
keithley.close()



sdev.ramp_zero()

InstrumentImporter.switch_utils.close(ser)

plt.figure()
plt.plot(np.arange(N), output*1e9, '-o')
plt.ylabel('output nA')
plt.xlabel('sample')