from instruments.DAC import IVVIrack
import time

# temporary imports
import numpy as np


x = np.matrix('1, 1, 1, 0, 0, 0; 1, 0, 0, 1, 1, 0; 0, 1, 0, 1, 0, 1; 0, 0, 1, 0, 1, 1')
inputvoltages = np.zeros(len(x))

def DacMeasure(x, Fs):
	for i in range(np.shape(x)[1]):
		for j in range(len(x)):
			inputvoltages[j] = x[j,i]
	
		IVVIrack.setinputVoltages(ivvi, inputVoltages)
		time.sleep(1)

	#measure and save to first part of voltage and time data.
