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
cf.nr_samples = 50
# cf.set_device = 'cdaq'

# Measure using the device specified in the config class.
keithley = Keithley2400.Keithley_2400('keithley', 'GPIB0::11')
keithley.compliancei.set(1E-6)
keithley.compliancev.set(4)
# keithley.nplci(0.01)   # set speed (fraction of powergrid frequency)
keithley.output.set(1)

ivvi = InstrumentImporter.IVVIrack.initInstrument(dac_step = 50, dac_delay = 0.001)

cf.filepath = r'D:\Lennart\iv_curves_ouput_2019-05-16\\'
cf.name = 'device6speedtestN200'
saveDirectory = SaveLib.createSaveDirectory(cf.filepath,cf.name)

output_data = np.zeros((7, cf.nr_samples))
all_input_data = np.zeros((7,cf.nr_samples,7))
static_values = np.linspace(-1000,1000, 7)
for ii in range(1,8):

    
    cf.switch_device = 6    # device number 1-8 on PCB
    cf.electrode = 6       # electrode which to apply voltage sweep to 1-8
    
    
    v_min = -2.0
    v_max = 2.0

    x = np.linspace(v_min, v_max, cf.nr_samples)*1000
    input_data = np.zeros((cf.nr_samples,7))
    input_data[:,cf.electrode-1] = x
    input_data[:, 3] = static_values[ii-1]
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
    
    start=time.time()
    for jj, single_input in enumerate(input_data):
        # sdev.ramp(single_input, ramp_speed=2.)
        InstrumentImporter.IVVIrack.setControlVoltages(ivvi, single_input)
        # time.sleep(0.5)
        output_data[ii-1,jj] = -keithley.curr()
    
    # sdev.ramp_zero(ramp_speed=2.)
    print('dev %i electrode %i took %0.5f sec' % (cf.switch_device, cf.electrode, time.time()-start))

    
    InstrumentImporter.switch_utils.close(ser)
    
    # plt.figure()
    # plt.plot(x, output_data*1e9, '-o')
    # plt.ylabel('Output (nA)')
    # plt.xlabel('Voltage on gate %i' %cf.electrode)
    # plt.title('Device #%i' % cf.switch_device)
    # plt.show()


keithley.output.set(0)
keithley.close()

output_data = output_data*1e9
SaveLib.saveArrays(saveDirectory, outputs=output_data, inputs=all_input_data)

plt.figure()
plt.plot(np.stack((x,)*7).T, output_data.T, marker='o')
plt.ylabel('Output (nA)')
plt.xlabel('Voltage applied (mV)')
plt.title('Device #%i' % cf.switch_device)
# plt.legend(['gate %i' % (i//4+i) for i in range(1,8)])
# plt.legend(static_values)
# plt.title('All devices, '+cf.name)
# plt.legend(['device %i' % i for i in range(1,9)])
plt.show()


InstrumentImporter.reset(0, 0)