# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:27:59 2019

@author: ljknoll
"""

from SkyNEt.instruments import InstrumentImporter
from SkyNEt.instruments.Keithley2400 import Keithley2400

import time
import numpy as np
import matplotlib.pyplot as plt


electrode = 7 # 1-7: 
N=20
input_data = np.zeros((N,7))
single_ramp_input = np.linspace(-0.5,1,N)*1000
input_data[:,electrode-1] = single_ramp_input
output = np.zeros(N)




ivvi = InstrumentImporter.IVVIrack.initInstrument()


ser = InstrumentImporter.switch_utils.init_serial('COM3')
InstrumentImporter.switch_utils.connect_single_device(ser, 7)

keithley = Keithley2400.Keithley_2400('keithley', 'GPIB0::11')
# Set compliances
keithley.compliancei.set(1E-6)
keithley.compliancev.set(4)
# Turn keithley output on
keithley.output.set(1)



start = time.time()
for ii,x in enumerate(input_data):
    InstrumentImporter.IVVIrack.setControlVoltages(ivvi, x)
    # Record current
#    time.sleep(0.05)
    output[ii] = -keithley.curr()

print( time.time()-start)

output = output*1e9

keithley.output.set(0)
keithley.close()

InstrumentImporter.switch_utils.close(ser)


plt.figure()
plt.plot(single_ramp_input, output, '-o')
plt.ylabel('output nA')
plt.xlabel('input voltage')
plt.show()


InstrumentImporter.reset(0, 0)