# from instruments.DAC import IVVIrack
import time
# temporary imports
import numpy as np


def DacMeasure(x, target, Fs):
	inputvoltages = np.zeros(len(x))
	singlemeas = target[0:len(target)/np.shape(x)[1]]
	output = np.zeros(len(target))

	for i in range(np.shape(x)[1]):
		for j in range(len(x)):
			inputvoltages[j] = x[j,i]
	
		IVVIrack.setinputVoltages(ivvi, inputVoltages)
		time.sleep(1)


		data = nidaqIO.IO(singlemeas, Fs)
		output[i*data+1:i*data+data] = data  # missing out on 0 but need the +1 to not remove the last digit of the first set do better.



	return output





	#measure and save to first part of voltage and time data.
