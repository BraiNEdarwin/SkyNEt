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
import SkyNEt.modules.SaveLib as SaveLib


electrode = 1 # 1-7: 
N=20
input_data = np.zeros((N,7))
single_ramp_input = np.linspace(-2,2,N)*1000
input_data[:,electrode-1] = single_ramp_input
output = np.zeros(N)


filepath = r'D:\Rik\IV-newdevices\\'
name = 'firsttest'
saveDirectory = SaveLib.createSaveDirectory(filepath,name)
connect_matrix = np.zeros((8,8))
connect_matrix[0,:] = [0, 0, 1, 1, 1, 1, 1, 0] ## C1 connects to e1 of D(fill in)
connect_matrix[1,:] = [0, 0, 1, 1, 1, 1, 1, 0] ## C2 connects to e2 of D(fill in)
connect_matrix[2,:] = [0, 0, 1, 1, 1, 1, 1, 0] ## C3 connects to e3 of D(fill in)
connect_matrix[3,:] = [0, 0, 1, 1, 1, 1, 1, 0] ## C4 connects to e4 of D(fill in)
connect_matrix[4,:] = [0, 0, 1, 1, 1, 1, 1, 0] ## C5 connects to e5 of D(fill in)
connect_matrix[5,:] = [0, 0, 1, 1, 1, 1, 1, 0] ## C6 connects to e6 of D(fill in)
connect_matrix[6,:] = [0, 0, 1, 1, 1, 1, 1, 0] ## C7 connects to e7 of D(fill in)
connect_matrix[7,:] = [0, 0, 1, 1, 1, 1, 1, 0] ## C8 connects to e8 of D(fill in


ivvi = InstrumentImporter.IVVIrack.initInstrument(comport = 'COM5')
for loop in range(3,8):
	ser = InstrumentImporter.switch_utils.init_serial('COM3')
	InstrumentImporter.switch_utils.connect_single_device(ser, loop)
	#InstrumentImporter.switch_utils.connect_matrix(ser,connect_matrix)
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


#InstrumentImporter.reset(0, 0)