# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:21:31 2019

@author: Darwin
"""

""" Copy of IV.py """

import matplotlib.pyplot as plt
from SkyNEt.instruments import InstrumentImporter
from SkyNEt.instruments.Keithley2400 import Keithley2400
import SkyNEt.modules.SaveLib as SaveLib
import numpy as np
import os
import time

class config:
    pass



cf = config()
cf.switch_comport = 'COM3'
cf.measure_device = 'keithley2400'
cf.nr_channels = 7
cf.nr_samples = 20
cf.set_device = 'cdaq'

# Measure using the device specified in the config class.
if cf.measure_device == 'keithley2400':
    keithley = Keithley2400.Keithley_2400('keithley', 'GPIB0::11')
    keithley.compliancei.set(1E-6)
    keithley.compliancev.set(4)
#    keithley.nplci(0.01)   # set speed (fraction of powergrid frequency)
    keithley.output.set(1)
else:
    raise('specify measurement device')

ivvi = InstrumentImporter.IVVIrack.initInstrument()

output_data = np.zeros((7, cf.nr_samples))
all_input_data = np.zeros((7,cf.nr_samples,7))
for ii in range(1,8):

    
    cf.switch_device = 7   # device number 1-8 on PCB
    cf.electrode = ii       # electrode which to apply voltage sweep to 1-8
    
    cf.filepath = r'D:\Lennart\iv_curves_to_output\\'
    cf.name = 'device7_ALL_electrodes-%i' % cf.electrode

    saveDirectory = SaveLib.createSaveDirectory(cf.filepath,cf.name)
    
    v_min = 0.
    v_max = 2.5

    x = np.linspace(v_min, v_max, cf.nr_samples)*1000
    input_data = np.zeros((cf.nr_samples,7))
    input_data[:,cf.electrode-1] = x
    all_input_data[ii-1] = input_data
    
    # arduino switch network
    # Initialize serial object
    ser = InstrumentImporter.switch_utils.init_serial(cf.switch_comport)
    # Switch to device
    InstrumentImporter.switch_utils.connect_single_device(ser, cf.switch_device)
    # Status print
    print('INFO: Connected device %i' % cf.switch_device)
    

    
    # set voltages with
    # if cf.set_device == 'cdaq':
    #     sdev = InstrumentImporter.nidaqIO.IO_cDAQ(nr_channels=cf.nr_channels)
    # else:
    #     raise('specify set device')
    
    
    for jj, single_input in enumerate(input_data):
        # sdev.ramp(single_input, ramp_speed=2.)
        InstrumentImporter.IVVIrack.setControlVoltages(ivvi, single_input)
        output_data[ii-1,jj] = -keithley.curr()
    
    # sdev.ramp_zero(ramp_speed=2.)
    

    
    InstrumentImporter.switch_utils.close(ser)
    
    # plt.figure()
    # plt.plot(x, output_data*1e9, '-o')
    # plt.ylabel('Output (nA)')
    # plt.xlabel('Voltage on gate %i' %cf.electrode)
    # plt.title('Device #%i' % cf.switch_device)
    # plt.show()


if cf.measure_device == 'keithley2400':
    keithley.output.set(0)
    keithley.close()

output_data = output_data*1e9
SaveLib.saveArrays(saveDirectory, outputs=output_data, inputs=all_input_data)

plt.figure()
plt.plot(output_data, '-o')
plt.ylabel('Output (nA)')
plt.xlabel('Voltage on gate %i' %cf.electrode)
plt.title('Device #%i' % cf.switch_device)
plt.show()


InstrumentImporter.reset(0, 0)