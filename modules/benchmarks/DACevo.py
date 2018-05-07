# from instruments.DAC import IVVIrack
import time
# temporary imports
import numpy as np
from instruments.niDAQ import nidaqIO
from instruments.DAC import IVVIrack


def DacMeasuremani(x, target, Fs):
	inputvoltages = np.zeros(len(x))
	singlemeas = target[0:len(target)/np.shape(x)[1]]
	output = np.zeros(len(target))

	for i in range(np.shape(x)[1]):
		for j in range(len(x)):
			inputvoltages[j] = x[j,i]
	
		IVVIrack.setinputVoltages(ivvi, inputVoltages)
		time.sleep(1)


		data = nidaqIO.IO(singlemeas, Fs)
		output[i*len(data):i*len(data)+len(data)-1] = data  


	return output



def DacMeasuredigit(x, target, Fs):
	x1 = np.matrix('0 0 0 0 0 1 1 0 1 1 1 0 0 0 1 0 0 0 0 1 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 1 1 1 0 1 1 0 1 1 1 1 1 1 0 1 1 1 1 0 0 1 1 0 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 0 0 1 1 1 0 ; 0 1 1 1 0 1 1 0 1 1 1 1 1 0 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 1 0 0 0 0 0 0 0 0 0 0; 0 1 1 1 0 1 1 0 1 1 1 1 0 1 1 1 1 1 1 0 1 1 1 0 1 1 1 1 1 0 0 1 1 1 0 1 1 1 0 1 0 1 1 1 0 1 1 1 1 0; 0 0 0 0 0 1 1 0 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 0 1 0 0 0 0 0 0 0 0 0 0 1 1 1 0 1 0 0 0 0 0 1 1 1 1 0')

	print(np.shape(x1)[1])

	for n in range(10):
		print(n)
		x = x1[:,n*5:n*5+5]
		for i in range(len(x)):
			inputvoltages = x[i,:]
			print(inputvoltages)
			# IVVIrack.setinputVoltages(ivvi, inputVoltages)
			# time.sleep(1)

			# measure using desire module.

			# output[i, n] = measureddata fill output in in array

		for j in range(len(x)):
			inputvoltages = x[:,j].T
			print(inputvoltages)
			# IVVIrack.setinputVoltages(ivvi, inputVoltages)
			# time.sleep(1)
	
			# # measure using desire module.

			# output[i+j+1, n] = measureddata 

