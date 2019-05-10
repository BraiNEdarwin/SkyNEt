# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:21:31 2019

@author: Darwin
"""

""" Copy of IV.py """

#import SkyNEt.modules.SaveLib as SaveLib
import matplotlib.pyplot as plt
from SkyNEt.instruments import InstrumentImporter
from SkyNEt.instruments.Keithley2400 import Keithley2400
import numpy as np
import os

class config:
    pass


cf = config()
cf.switch_comport = 'COM3'
cf.measure_device = 'keithley2400'
cf.nr_channels = 7
cf.nr_samples = 20

cf.switch_device = 0    # device number (0-7)
cf.electrode = 4        # electrode which to apply voltage sweep to

# sweep from v_min to v_max
v_min = -1
v_max = 1

x = np.linspace(v_min, v_max, cf.nr_samples)
input_data = np.zeros((cf.nr_samples,7))
input_data[:,cf.electrode] = x

# arduino switch network
# Initialize serial object
ser = InstrumentImporter.switch_utils.init_serial(cf.switch_comport)
# Switch to device
InstrumentImporter.switch_utils.connect_single_device(ser, cf.switch_device)
# Status print
print('INFO: Connected device %i' % cf.switch_device)

# Measure using the device specified in the config class.
if cf.measure_device == 'keithley2400':
    keithley = Keithley2400.Keithley_2400('keithley', 'GPIB0::11')
    keithley.compliancei.set(1E-7)
    keithley.compliancev.set(4)
#    keithley.nplci(0.01)   # set speed (fraction of powergrid frequency)
    keithley.output.set(1)
else:
    raise('specify measurement device')

# set voltages with
if cf.set_device == 'cdaq':
    sdev = InstrumentImporter.nidaqIO.IO_cDAQ(nr_channels=cf.nr_channels)
else:
    raise('specify set device')

output = np.zeros(cf.nr_samples)
for ii, single_input in enumerate(input_data):
    sdev.ramp(single_input)
    output[ii] = keithley.curr()

sdev.ramp_zero()


if cf.measure_device == 'keithley2400':
    keithley.output.set(0)
    keithley.close()

InstrumentImporter.switch_utils.close(ser)

plt.figure()
plt.plot(x, output*1e9, '-o')
plt.ylabel('Output (nA)')
plt.xlabel('Voltage on gate %i' %cf.electrode)
